
import os
import math

from pathlib import Path
import hashlib, fcntl

from tqdm import tqdm

import torch
import torch.distributed as dist

from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, DistributedSampler

from Total_Dataset import collate_fn

# =====================================
# 1️⃣ Action Expert (Temporal Decoder)
# =====================================
def build_vl_cache_distributed_optimized(
    model,
    dataset,
    device="cuda",
    *,
    batch_size=512,          # DataLoader 배치 (VRAM 24GB면 2~4 권장)
    num_workers=8,
    prefetch_factor=4,
    micro_bs=2,            # 마이크로 배치 (OOM 시 자동 백오프)
    key_mode="full",       # "traj": traj_path 단위 캐시 / "full": (traj+lang+views) 단위
    rank_sharded_cache=False,  # rank별 캐시 폴더 분리
    cache_dir_fallback="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
):
    """
    고속/안정 캐싱:
      - 마이크로배칭 + OOM 백오프
      - use_cache=False (KV cache 비활성화)
      - rank별 캐시 폴더 분리로 fcntl 락 경합 최소화
      - tqdm 진행률, miss/skipped 통계 표시
      - key_mode="traj"로 두면 동일 traj는 1회만 계산 (속도 10~30배↑)

    model 요구사항:
      - model.vl_model, model.processor 필요
      - (선택) model.cache_dir 있으면 사용, 없으면 cache_dir_fallback 사용
      - (선택) model._atomic_save / model._enforce_cache_limit 있으면 사용
    """

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # base cache dir
    base_cache_dir = getattr(model, "cache_dir", None)
    if base_cache_dir is None:
        base_cache_dir = Path(cache_dir_fallback)
    else:
        base_cache_dir = Path(base_cache_dir)

    # rank-sharded dir (락 경쟁 최소화)
    target_cache_dir = base_cache_dir / (f"rank_{rank}" if rank_sharded_cache else "")
    target_cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # 로컬 헬퍼: 안전 저장/경로 생성
    # ---------------------------
    def _local_atomic_save(tensor_cpu: torch.Tensor, path: Path):
        """model에 _atomic_save가 있으면 사용, 없으면 로컬 구현"""
        if hasattr(model, "_atomic_save"):
            return model._atomic_save(tensor_cpu, path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        lock_path = str(path) + ".lock"
        with open(lock_path, "w") as lockfile:
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX)
                if path.exists():
                    return
                torch.save(tensor_cpu, tmp)
                os.replace(tmp, path)
            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)

    def _local_enforce_cache_limit(max_gb=40):
        """model에 _enforce_cache_limit 있으면 사용, 없으면 로컬로 target_cache_dir만 정리"""
        if hasattr(model, "_enforce_cache_limit"):
            try:
                # model이 자기 cache_dir을 관리한다면 호출
                model._enforce_cache_limit(max_gb=max_gb)
                return
            except TypeError:
                pass  # 시그니처 차이 무시하고 로컬 처리로 폴백
        total_bytes = 0
        files = []
        for f in target_cache_dir.glob("*.pt"):
            total_bytes += f.stat().st_size
            files.append(f)
        limit = max_gb * (1024 ** 3)
        if total_bytes > limit:
            files = sorted(files, key=lambda f: f.stat().st_mtime)
            while total_bytes > limit and files:
                f = files.pop(0)
                total_bytes -= f.stat().st_size
                f.unlink(missing_ok=True)

    def _cache_path_for(traj_key: str, txt: str, views: list[str | None]) -> Path:
        """key_mode에 따라 캐시 키 구성."""
        if key_mode == "traj":
            # traj_path만 기준 (같은 traj는 1회만 계산)
            base = traj_key.split("::t=")[0]
        else:
            # 기존 방식: traj+lang+views 모두 반영
            vlist = [v for v in views if v is not None]
            base = traj_key + "||" + txt + "||" + "|".join(vlist)
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]
        return target_cache_dir / f"{h}.pt"

    # ---------------------------
    # DataLoader (샘플 분배 보장)
    # ---------------------------
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )

    total_local = math.ceil(len(dataset) / world_size)
    print(f"[Rank {rank}] Assigned ~{total_local} samples for caching.")
    print(f"[Rank {rank}] CUDA ready: {torch.cuda.is_available()}, device={torch.cuda.current_device()}")

    # ---------------------------
    # 캐싱 루프
    # ---------------------------
    if hasattr(model, "eval"):
        model.eval()

    total_cached, total_skipped, total_processed = 0, 0, 0
    pbar = tqdm(
        total=total_local,
        desc=f"[Rank {rank}] Caching progress",
        dynamic_ncols=True,
        disable=(rank != 0)
    )

    with torch.inference_mode():
        for batch_idx, batch in enumerate(data_loader):
            texts = batch["instruction"]
            image_paths_list = batch["images"]
            keys = batch["cache_keys"]

            # --- 미스/스킵 분리 ---
            miss_items = []
            for key, txt, views in zip(keys, texts, image_paths_list):
                cpath = _cache_path_for(key, txt, views)
                if not cpath.exists():
                    miss_items.append({"text": txt, "views": views, "key": key, "cpath": cpath})
                else:
                    total_skipped += 1

            total_processed += len(keys)
            if not miss_items:
                pbar.update(len(keys))
                if rank == 0:
                    cached_ratio = (total_cached / max(1, total_processed)) * 100
                    pbar.set_postfix({
                        "cached": total_cached,
                        "skipped": total_skipped,
                        "miss%": f"{100 - cached_ratio:.1f}%",
                        "GPU": f"{torch.cuda.memory_allocated(device)/1e9:.1f}GB"
                    })
                continue

            # --- 메시지 전처리 (CPU) ---
            messages_list = []
            for item in miss_items:
                txt, views = item["text"], item["views"]
                msg_content = [{"type": "image", "image": v} for v in views if v is not None]
                msg_content.append({"type": "text", "text": txt})
                messages_list.append([{"role": "user", "content": msg_content}])

            processed_texts, vision_inputs_list = [], []
            for messages in messages_list:
                text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                vision_inputs, _ = process_vision_info(messages)
                processed_texts.append(text)
                vision_inputs_list.append(vision_inputs)

            # --- 마이크로배칭 + OOM 백오프 ---
            start = 0
            _micro_bs = max(1, micro_bs)
            while start < len(miss_items):
                end = min(start + _micro_bs, len(miss_items))
                sub_items  = miss_items[start:end]
                sub_texts  = processed_texts[start:end]
                sub_vision = vision_inputs_list[start:end]

                try:
                    inputs = model.processor(
                        text=sub_texts,
                        images=sub_vision,
                        padding=True,
                        return_tensors="pt"
                    ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

                    outputs = model.vl_model(
                        **inputs,
                        output_hidden_states=True,
                        use_cache=False,          # ✅ 메모리 절감
                        return_dict=True
                    )
                    vl_tokens_batch = outputs.hidden_states[-1]
                    pooled_batch = vl_tokens_batch.mean(dim=1, keepdim=True)

                    for j, item in enumerate(sub_items):
                        pooled_single = pooled_batch[j:j+1]
                        _local_atomic_save(
                            pooled_single.detach().to("cpu", dtype=torch.float16),
                            item["cpath"]
                        )
                        total_cached += 1

                    # 정리
                    del inputs, outputs, vl_tokens_batch, pooled_batch
                    torch.cuda.empty_cache()

                    start = end  # 다음 마이크로 배치로 진행

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        torch.cuda.empty_cache()
                        if _micro_bs == 1:
                            raise  # 더 줄일 수 없음
                        _micro_bs = max(1, _micro_bs // 2)
                        if rank == 0:
                            print(f"[OOM] Lowering micro_bs to #{_micro_bs} and retrying...")
                        continue
                    else:
                        raise

            # --- 진행률 업데이트 ---
            pbar.update(len(keys))
            if rank == 0:
                cached_ratio = (total_cached / max(1, total_processed)) * 100
                pbar.set_postfix({
                    "cached": total_cached,
                    "skipped": total_skipped,
                    "miss%": f"{100 - cached_ratio:.1f}%",
                    "GPU": f"{torch.cuda.memory_allocated(device)/1e9:.1f}GB"
                })

            # (선택) 주기적 캐시 용량 제한
            _local_enforce_cache_limit(max_gb=40)

    pbar.close()
    print(f"[Rank {rank}] ✅ Finished. Cached {total_cached} / Skipped {total_skipped}")
    dist.barrier()
    if rank == 0:
        print("🚀 All ranks finished caching. Cache is ready for training.")
