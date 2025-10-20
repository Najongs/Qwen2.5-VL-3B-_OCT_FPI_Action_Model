import argparse
import wandb
import io, shutil, threading, queue, time
import os
import re
import math
import glob
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data import random_split

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_float32_matmul_precision("high")

from model import QwenActionExpert, Not_freeze_QwenVLAForAction
from Total_Dataset import collate_fn, infer_lang_from_path
from Total_Dataset import BridgeRawSequenceDataset, insertionMeca500Dataset
from Make_VL_cache import build_vl_cache_distributed_optimized

# ======== I/O & Checkpoint Utils ========
STAGING_DIR = Path("/dev/shm/qwen_vla_stage")   # 로컬 RAM/NVMe (없으면 /tmp 권장)
CKPT_DIR     = Path("./checkpoints")            # NAS 또는 공유 디렉토리
STAGING_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def _atomic_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if src != tmp:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)

def copy_to_local_then_load(src_path: Path, map_location):
    """네트워크 파일을 로컬 스테이징으로 빠르게 복사 후 torch.load"""
    if not src_path.exists():
        raise FileNotFoundError(str(src_path))
    local_copy = STAGING_DIR / src_path.name
    shutil.copy2(src_path, local_copy)  # 보통 이 경로가 훨씬 빠름
    # PyTorch 2.4+면 weights_only=True가 빠르고 안전
    try:
        return torch.load(local_copy, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(local_copy, map_location=map_location)

class AsyncCheckpointWriter:
    """학습은 그대로 진행, 저장은 백그라운드 스레드가 처리"""
    def __init__(self, max_queue=2, sync_every=0):
        self.q = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.stop = False
        self.sync_every = sync_every  # 0이면 즉시 처리
        self.thread.start()

    def _worker(self):
        last_sync = time.time()
        while not self.stop:
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            state_dict, final_dst = payload["state"], Path(payload["dst"])
            # 1) 로컬 스테이징에 먼저 저장 (CPU 텐서)
            local_tmp = STAGING_DIR / (final_dst.name + f".{int(time.time())}.pt")
            # 모델/옵티마이저 텐서를 CPU로 복사한 상태를 받는 것이 이상적
            torch.save(state_dict, local_tmp, _use_new_zipfile_serialization=True)
            # 2) 필요 시 배치/주기 동기화
            if self.sync_every > 0 and (time.time() - last_sync) < self.sync_every:
                continue
            # 3) 원자적 교체로 최종 목적지에 반영
            _atomic_move(local_tmp, final_dst)
            last_sync = time.time()

    def submit(self, state_dict, final_dst: Path):
        # 큐가 가득 차 있으면 가장 오래된 걸 버리고 최신으로 교체(학습 지연 방지)
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put({"state": state_dict, "dst": str(final_dst)})

    def close(self):
        self.stop = True
        self.thread.join(timeout=5)

def build_trapezoid_scheduler(
    optimizer,
    total_steps: int,
    *,
    base_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_ratio: float = 0.03,
    hold_ratio: float = 0.02,
):
    """
    LLM 스타일: Warmup -> Hold(선택) -> Cosine Decay
    - warmup_ratio: 전체 step 대비 워밍업 비율
    - hold_ratio  : 워밍업 후 고정 유지 비율 (0 가능)
    - 나머지는 cosine으로 min_lr까지 감쇠
    """
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps   = int(total_steps * hold_ratio)
    decay_steps  = max(1, total_steps - warmup_steps - hold_steps)
    floor = min_lr / max(base_lr, 1e-12)

    def lr_lambda(step: int):
        if step < warmup_steps:
            # 선형 워밍업: 0 -> 1
            return (step + 1) / max(1, warmup_steps)
        elif step < warmup_steps + hold_steps:
            # 유지: 1.0
            return 1.0
        else:
            # 코사인 감쇠: 1 -> floor
            t = (step - warmup_steps - hold_steps) / decay_steps
            t = min(max(t, 0.0), 1.0)
            cos = 0.5 * (1 + math.cos(math.pi * t))  # 1 -> 0
            return floor + (1 - floor) * cos

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def build_rewarm_scheduler(
    optimizer,
    total_steps: int,
    *,
    prev_lr: float,        # 마지막 학습에서 쓰던 실 lr (예: 1e-8)
    target_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_ratio: float = 0.05,
    hold_ratio: float = 0.05,
):
    """
    🔁 ReWarm Scheduler (절대 lr 스케줄을 factor로 표현)
      - base_lr = target_lr 로 잡고
      - factor를 prev_lr/target_lr → 1.0 으로 warmup
      - hold 후 1.0 → (min_lr/target_lr) 로 cosine decay
    """
    assert target_lr > 0 and min_lr > 0
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps   = int(total_steps * hold_ratio)
    decay_steps  = max(1, total_steps - warmup_steps - hold_steps)

    floor = min_lr / target_lr                 # decay 최하 한계 (factor)
    start = max(1e-12, prev_lr / target_lr)    # warmup 시작 factor(아주 작을 수 있음)

    def lr_lambda(step: int):
        if step < warmup_steps:
            # 선형: start → 1.0
            prog = (step + 1) / max(1, warmup_steps)
            return start + (1.0 - start) * prog
        elif step < warmup_steps + hold_steps:
            return 1.0
        else:
            # cosine: 1.0 → floor
            t = (step - warmup_steps - hold_steps) / decay_steps
            t = min(max(t, 0.0), 1.0)
            cos = 0.5 * (1 + math.cos(math.pi * t))   # 1 → 0
            return floor + (1.0 - floor) * cos

    # ① base_lr를 target_lr로 맞춤
    for g in optimizer.param_groups:
        g["lr"] = target_lr

    sched = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ② 첫 step 전에 실 lr을 prev_lr로 맞춰 시작 (warmup 시작점)
    for g in optimizer.param_groups:
        g["lr"] = prev_lr
    return sched

# ===========================================================
# 1️⃣ 초기화
# ===========================================================
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank

# ===========================================================
# 2️⃣ 학습 루프
# ===========================================================
def Train(
    model,
    data_loader,
    optimizer,
    num_epochs=3,
    grad_accum_steps=8,
    device="cuda",
    save_path="./checkpoints/qwen_vla.pt",
    scheduler=None,
    sched_on="step",
    val_loader=None,
    start_epoch=0,           # ✅ 추가
):
    loss_fn = nn.MSELoss()
    rank = dist.get_rank()
    writer = AsyncCheckpointWriter(max_queue=2, sync_every=0) if rank == 0 else None

    model.train()
    if rank == 0:
        wandb.init(
            project="QwenVLA",
            name=f"train_run_{time.strftime('%m%d_%H%M')}",
            resume="allow",
            id=f"qvla_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum_steps": grad_accum_steps,
                "epochs": num_epochs,
                "scheduler": sched_on,
            }
        )

    global_step = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        optimizer.zero_grad()
        model.train()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f"[Rank {rank}] Epoch {epoch+1}",
                    disable=(rank != 0))

        for step, batch in pbar:
            instructions = batch["instruction"]
            image_inputs = batch["images"]
            gt_actions = batch["actions"].to(device, dtype=torch.bfloat16)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_actions, _ = model(
                    text_inputs=instructions,
                    image_inputs=image_inputs,
                    z_chunk=gt_actions,
                    cache_keys=batch["cache_keys"],
                )

                weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)
                weights = weights / weights.mean()
                loss_each = (pred_actions.float() - gt_actions.float()).pow(2).mean(dim=[1,2])
                loss = (loss_each * weights).mean() / grad_accum_steps

            loss.backward()
            total_loss += loss.item() * grad_accum_steps

            # === gradient step ===
            if (step + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None and sched_on == "step":
                    scheduler.step()

                global_step += 1

                # === ✅ tqdm 및 wandb 로깅 ===
                lr = optimizer.param_groups[0]["lr"]
                if rank == 0:
                    pbar.set_postfix({
                        "loss": f"{loss.item() * grad_accum_steps:.6f}",
                        "lr": f"{lr:.2e}",
                        "grad": f"{grad_norm:.2f}"
                    })
                    wandb.log({
                        "train/loss_step": loss.item() * grad_accum_steps,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "global_step": global_step
                    })

        # === epoch 평균 ===
        avg_loss_tensor = torch.tensor(total_loss / len(data_loader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

        # === scheduler per epoch ===
        if scheduler is not None and sched_on == "epoch":
            scheduler.step()

        # === ✅ Validation Loop ===
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_sum, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    gt_actions = batch["actions"].to(device, dtype=torch.bfloat16)
                    pred_actions, _ = model(
                        text_inputs=batch["instruction"],
                        image_inputs=batch["images"],
                        z_chunk=gt_actions,
                        cache_keys=batch["cache_keys"],
                    )
                    weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)
                    weights = weights / weights.mean()  # 평균 1로 정규화
                    loss_each = (pred_actions.float() - gt_actions.float()).pow(2).mean(dim=[1,2])  # 샘플별 MSE
                    loss = (loss_each * weights).mean() / grad_accum_steps
                    val_loss_sum += loss.item()
                    val_count += 1
            val_loss = val_loss_sum / max(1, val_count)
            model.train()


    
        # === epoch 종료 후 ===
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            frozen = total_params - trainable

            import psutil, gc
            gpu_mem = torch.cuda.memory_allocated()/1e9
            cpu_mem = psutil.virtual_memory().percent
            gc.collect()

            # === 추가 로깅 ===
            wandb.log({
                "epoch": epoch + 1,
                "train/loss_epoch": avg_loss,
                "val/loss_epoch": val_loss if val_loss else None,
                "params/trainable_M": trainable / 1e6,
                "params/frozen_M": frozen / 1e6,
                "params/frozen_ratio": frozen / total_params,
                "system/gpu_mem_GB": gpu_mem,
                "system/cpu_mem_%": cpu_mem,
                "lr/base_lr": optimizer.param_groups[0]["lr"],
                "lr/vl_lr": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None,
                "lr/vision_lr": optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else None,
                "scheduler/phase_ratio_warmup": scheduler._get_lr_lambda(0) if hasattr(scheduler, "_get_lr_lambda") else None,
            })

            print(f"[DEBUG] GPU {gpu_mem:.2f} GB / CPU {cpu_mem:.1f}% used "
                  f"| Trainable {trainable/1e6:.2f}M / Frozen {frozen/1e6:.2f}M")

            # === LoRA 파라미터 로깅 (선택) ===
            lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n}
            if lora_params:
                avg_abs = np.mean([p.data.abs().mean().item() for p in lora_params.values()])
                wandb.log({"lora/avg_weight_abs": avg_abs})

            print(f"\n📊 Epoch {epoch+1} Summary | Train: {avg_loss:.8f} | "
                  f"Val: {val_loss:.8f}" if val_loss else f"\n📊 Epoch {epoch+1} Train Loss: {avg_loss:.8f}")

            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            base_path = Path(save_path)

            # === Best model 저장 여부 판단 ===
            if not hasattr(Train, "_best_loss"):
                Train._best_loss = float("inf")

            is_best = val_loss is not None and val_loss < Train._best_loss

            if is_best:
                Train._best_loss = val_loss
                best_path = save_dir / "qwen_vla_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "val_loss": val_loss,
                }, best_path)
                print(f"🏆 [Best] Validation improved → saved to {best_path}")

            else:
                # Best가 아닌 경우 기존 latest만 덮어쓰기
                tmp_path = base_path.with_suffix(".tmp")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "val_loss": val_loss,
                }, tmp_path)
                os.replace(tmp_path, base_path)
                print(f"💾 [Sync] Latest checkpoint updated: {base_path}")

    if rank == 0 and writer is not None:
        writer.close()

    if rank == 0:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    # ===== 기존 VLA 기본 학습 옵션 =====
    parser.add_argument("--mode", choices=["cache", "train"], required=True,
                        help="Mode: 'cache' to build feature cache, 'train' to train action expert")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--hold-ratio", type=float, default=0.02)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--sched-on", choices=["step", "epoch"], default="step")

    # ===== 추가된 LoRA / Fine-tuning 옵션 =====
    parser.add_argument("--finetune-vl", choices=["none", "lora", "full"], default="lora",
                        help="Qwen-VL fine-tuning mode")
    parser.add_argument("--vl-lr", type=float, default=1e-5,
                        help="learning rate for VL backbone (LoRA/full mode)")
    parser.add_argument("--vision-lr", type=float, default=5e-6,
                        help="learning rate for vision encoder if full fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--unfreeze-last-n", type=int, default=2,
                        help="number of transformer blocks to unfreeze when full fine-tuning")

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"🚀 [Rank {rank}] Running in {args.mode.upper()} mode on {world_size} GPUs")

    # # === 1️⃣ BRIDGE 데이터셋 ===
    # bridge_root = "/home/najo/NAS/VLA/dataset/raw"
    # bridge_ds = BridgeRawSequenceDataset(root=bridge_root, horizon=8)

    # === 2️⃣ Meca500 (OCT Insertion) ===
    json_file_path_list = [
        f"/home/najo/NAS/VLA/dataset/OCT_insertion/Captures{i}/Captures{i}_precise_9views.json"
        for i in range(1, 8)
    ]
    meca_datasets = [insertionMeca500Dataset(json_path=p, horizon=8) for p in json_file_path_list]
    meca_ds = ConcatDataset(meca_datasets)

    # === 3️⃣ ZED 기반 추가 파트 ===
    zed_paths = []
    for i in range(1, 21):
        part = "part1" if i <= 10 else "part2"
        zed_paths.append(f"/home/najo/NAS/VLA/dataset/{part}/ZED_Captures_{i}th/ZED_Captures_{i}th_precise_8views.json")

    zed_datasets = [insertionMeca500Dataset(json_path=p, horizon=8) for p in zed_paths]
    zed_ds = ConcatDataset(zed_datasets)

    # === 4️⃣ 전체 통합 ===
    dataset = ConcatDataset([meca_ds, zed_ds]) # bridge_ds, 

    if rank == 0:
        print(f"✅ Unified dataset size: {len(dataset)} samples.")

    
    if rank == 0:
        print(f"✅ [Rank {rank}] Dataset initialized with {len(dataset)} samples.")

    vl_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # ===========================================================
    # 3️⃣ 캐시 생성 모드
    # ===========================================================
    if args.mode == "cache":
        if rank == 0:
            print("⏳ Initializing VL-only model for cache building...")

        processor = AutoProcessor.from_pretrained(vl_model_name)
        vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vl_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            low_cpu_mem_usage=True,          # ← 로딩 속도/메모리 최적화
        )

        class DummyVLA:
            def __init__(self, vl_model, processor):
                self.vl_model = vl_model
                self.processor = processor
                self.cache_dir = Path("/home/najo/NAS/VLA/dataset/cache/qwen_vl_features")
                self.cache_dir.mkdir(parents=True, exist_ok=True)

                # ✅ 인스턴스 메서드만 바인딩
                self._cache_path = Not_freeze_QwenVLAForAction._cache_path.__get__(self)
                self._enforce_cache_limit = Not_freeze_QwenVLAForAction._enforce_cache_limit.__get__(self)

                # ✅ staticmethod는 그대로 복사
                self._atomic_save = Not_freeze_QwenVLAForAction._atomic_save

            def eval(self):
                self.vl_model.eval()
                return self

        dummy_model = DummyVLA(vl_model, processor)

        # 캐시 빌드
        build_vl_cache_distributed_optimized(
            dummy_model, dataset, device=device,
            rank_sharded_cache=False
        )
        
        dist.barrier()
        if rank == 0:
            print("✅ Cache build complete. You can now run training with --mode train.")
        dist.destroy_process_group()
        return  # 캐시 후 종료

    # ===========================================================
    # 4️⃣ 학습 모드
    # ===========================================================
    if args.mode == "train":
        if rank == 0:
            print("⏳ Initializing full QwenVLA model for training...")

        model = Not_freeze_QwenVLAForAction(
            vl_model_name=vl_model_name,
            action_dim=7,
            horizon=8,
            hidden_dim=1024,
            finetune_vl=args.finetune_vl,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            unfreeze_last_n=args.unfreeze_last_n
        ).to(device)
        
        # 전체 데이터셋 분할
        total_len = len(dataset)
        val_len = int(total_len * 0.05)   # 5% validation
        train_len = total_len - val_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])

        # DDP용 Sampler
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

        # 각각에 대한 DataLoader
        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            num_workers=4,
            sampler=train_sampler,
            collate_fn=collate_fn,
            prefetch_factor=4,
            persistent_workers=False,
            pin_memory=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=4,
            sampler=val_sampler,
            collate_fn=collate_fn,
            persistent_workers=False,
            pin_memory=False,
        )

        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        
        # === Optimizer 구성 ===
        def wd_filter(name, param):
            if param.ndim == 1: return False
            if name.endswith(".bias"): return False
            return True

        ae_named = list(model.module.action_expert.named_parameters())
        vl_named = list(model.module.vl_model.named_parameters())

        ae_decay    = [p for n,p in ae_named if wd_filter(n,p) and p.requires_grad]
        ae_n_decay  = [p for n,p in ae_named if not wd_filter(n,p) and p.requires_grad]

        vl_decay   = [p for n,p in vl_named if wd_filter(n,p) and p.requires_grad]
        vl_n_decay = [p for n,p in vl_named if not wd_filter(n,p) and p.requires_grad]

        vision_decay, vision_n_decay = [], []
        for n,p in vl_named:
            if not p.requires_grad:
                continue
            if "vision" in n or "visual" in n or "vision_tower" in n:
                (vision_decay if wd_filter(n,p) else vision_n_decay).append(p)

        # === LR 설정 ===
        param_groups = [
            {"params": ae_decay,        "lr": args.lr,      "weight_decay": 0.01},
            {"params": ae_n_decay,      "lr": args.lr,      "weight_decay": 0.0},
        ]

        if args.finetune_vl == "lora":
            param_groups += [
                {"params": vl_decay,    "lr": args.vl_lr,   "weight_decay": 0.01},
                {"params": vl_n_decay,  "lr": args.vl_lr,   "weight_decay": 0.0},
            ]
        elif args.finetune_vl == "full":
            param_groups += [
                {"params": vision_decay,    "lr": args.vision_lr,  "weight_decay": 0.01},
                {"params": vision_n_decay,  "lr": args.vision_lr,  "weight_decay": 0.0},
                {"params": vl_decay,        "lr": args.vl_lr,      "weight_decay": 0.01},
                {"params": vl_n_decay,      "lr": args.vl_lr,      "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

        
        # =======================================================
        # ✅ 체크포인트 복원 (model+optim) + epoch 복원
        # =======================================================
        ckpt_path = "./checkpoints/qwen_vla_final_1000.pt"
        start_epoch = 0
        if os.path.exists(ckpt_path):
            if rank == 0:
                print(f"🔄 Found checkpoint at {ckpt_path}, resuming training...")
            checkpoint = copy_to_local_then_load(Path(ckpt_path), map_location=device)

            try:
                model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print("✅ Loaded model weights (partial, strict=False)")
            except KeyError:
                model.module.load_state_dict(checkpoint, strict=False)


            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except ValueError:
                    print("⚠️ Optimizer group mismatch detected — skipping optimizer state load.")


            # ✅ 여기서 epoch 반드시 복원
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                if rank == 0:
                    print(f"✅ Resumed from epoch {start_epoch}")
        else:
            if rank == 0:
                print("🆕 No checkpoint found, starting from scratch.")

        # ===== 남은 epoch/step 계산 =====
        total_epochs = 1100
        remaining_epochs = max(1, total_epochs - start_epoch)
        iters_per_epoch = len(train_loader)
        steps_per_epoch = math.ceil(iters_per_epoch / max(1, args.grad_accum_steps))
        total_steps = steps_per_epoch * remaining_epochs

        # ===== 스케줄러 설정 =====
        current_lr = optimizer.param_groups[0]["lr"]   # ckpt에서 불러온 현재 lr (ex. 1e-8)
        target_lr  = args.lr

        if start_epoch > 0:
            # 이어 달리기 → prev_lr(=current_lr)에서 target_lr로 rewarm
            scheduler = build_rewarm_scheduler(
                optimizer,
                total_steps=total_steps,
                prev_lr=current_lr,
                target_lr=target_lr,
                min_lr=args.min_lr,
                warmup_ratio=0.05,
                hold_ratio=0.05,
            )
            if rank == 0:
                print(f"🔁 [ReWarm] {current_lr:.2e} → {target_lr:.2e}, then cosine → {args.min_lr:.2e}")
        else:
            # 처음 시작 → 일반 trapezoid
            scheduler = build_trapezoid_scheduler(
                optimizer,
                total_steps=total_steps,
                base_lr=target_lr,
                min_lr=args.min_lr,
                warmup_ratio=args.warmup_ratio,
                hold_ratio=args.hold_ratio,
            )
            if rank == 0:
                print(f"🆕 [New] 0 → {target_lr:.2e} (warmup/hold/cosine)")

        save_path = './checkpoints/qwen_vla_train.pt'
        # ==================================F=====================
        # ✅ Train 호출 (남은 epoch만큼 진행)
        # =======================================================
        Train(
            model,
            train_loader,
            optimizer,
            num_epochs=remaining_epochs,
            grad_accum_steps=args.grad_accum_steps,
            device=device,
            save_path=save_path,
            scheduler=scheduler,
            sched_on=args.sched_on,
            val_loader=val_loader,
            start_epoch=start_epoch
        )

        # =======================================================
        # ✅ 안전 저장 보장 (Async writer flush)
        # =======================================================
        if rank == 0 and "writer" in locals() and writer is not None:
            print("💾 Flushing async checkpoints to disk...")
            writer.flush()
            writer.close()
            print("✅ All pending checkpoints flushed safely.")

        # ✅ 최종 체크포인트 저장 (모든 상태 포함)
        if rank == 0:
            final_path = Path("./checkpoints/qwen_vla_final.pt")
            torch.save({
                "epoch": total_epochs - 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            }, final_path)
            print(f"✅ [Final] Final checkpoint saved at {final_path}")

        dist.destroy_process_group()
        if rank == 0:
            print("🧹 DDP process group destroyed. Training complete.")

if __name__ == "__main__":
    os.environ["PYTHONBUFFERED"] = "1"
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
