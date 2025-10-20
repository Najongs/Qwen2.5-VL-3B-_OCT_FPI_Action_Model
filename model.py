import os
from pathlib import Path
import hashlib, fcntl
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model

from concurrent.futures import ThreadPoolExecutor

class QwenActionExpert(nn.Module):
    def __init__(self, vl_dim=3072, action_dim=7, horizon=8,
                 hidden_dim=1024, nhead=8, num_layers=4):
        super().__init__()
        self.horizon = horizon
        self.cond_proj = nn.Linear(vl_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, hidden_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.temporal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, vl_tokens: torch.Tensor, z_chunk: torch.Tensor):
        B, H, A = z_chunk.shape
        cond = self.cond_proj(vl_tokens.mean(dim=1, keepdim=True))  # (B,1,Hd)
        tgt = self.pos_embed.repeat(B, 1, 1)                        # (B,H,Hd)
        decoded = self.temporal_decoder(tgt, cond)                  # (B,H,Hd)
        delta = self.output_head(decoded)                           # (B,H,A)
        pred_actions = z_chunk + delta
        return pred_actions, delta

# =====================================
# 2️⃣ Full Vision-Language-Action Model
# =====================================
class QwenVLAForAction(nn.Module):
    def __init__(self,
                 vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 action_dim=7, horizon=8, hidden_dim=1024,
                 cache_dir="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features"):
        super().__init__()
        print(f"🚀 Loading Qwen-VL backbone: {vl_model_name}")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # ✅ 디렉토리 생성

        self.processor = AutoProcessor.from_pretrained(vl_model_name)
        self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vl_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            low_cpu_mem_usage=True,
        )

        self.action_expert = QwenActionExpert(
            vl_dim=self.vl_model.config.hidden_size,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim
        ).to(dtype=torch.bfloat16, device="cuda")

        # Freeze base model
        print("🧊 Freezing Qwen-VL parameters...")
        for p in self.vl_model.parameters():
            p.requires_grad = False
        print("✅ Frozen.")

    def _cache_path(self, key: str, txt: str, views: list[str | None]) -> Path:
        vlist = [v for v in views if v is not None]
        raw = key + "||" + txt + "||" + "|".join(vlist)
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{h}.pt"

    @staticmethod
    def _atomic_save(tensor_cpu: torch.Tensor, path: Path):
        tmp = path.with_suffix(".pt.tmp")
        with open(str(path) + ".lock", "w") as lockfile:
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX)
                if path.exists():
                    return  # 이미 다른 rank가 저장 완료한 경우
                torch.save(tensor_cpu, tmp)
                os.replace(tmp, path)
            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)

    def _enforce_cache_limit(self, max_gb=20):
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt"))
        if total_bytes > max_gb * (1024 ** 3):
            all_files = sorted(self.cache_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
            while total_bytes > max_gb * (1024 ** 3) and all_files:
                f = all_files.pop(0)
                total_bytes -= f.stat().st_size
                f.unlink(missing_ok=True)
            print(f"⚠️ Cache limit exceeded. Trimmed to {max_gb}GB.")

    def forward(self, text_inputs, image_inputs, z_chunk, cache_keys=None):
        device = next(self.parameters()).device

        if cache_keys is None:
            cache_keys = [f"idx={i}" for i in range(len(text_inputs))]

        pooled_vl_tokens_dict = {}

        # --- 1️⃣ 캐시 HIT ---
        miss_items = []
        for txt, views, key in zip(text_inputs, image_inputs, cache_keys):
            cache_path = self._cache_path(key, txt, views)
            if cache_path.exists():
                pooled = torch.load(cache_path, map_location="cpu")
                pooled = pooled.pin_memory().to(device=device, non_blocking=True, dtype=torch.bfloat16)
                pooled_vl_tokens_dict[key] = pooled
            else:
                miss_items.append((txt, views, key))

        # --- 2️⃣ 캐시 MISS: 병렬 전처리 ---
        if miss_items:
            def preprocess_message(args):
                txt, views, key = args
                msg_content = [{"type": "image", "image": v} for v in views if v is not None]
                msg_content.append({"type": "text", "text": txt})
                messages = [{"role": "user", "content": msg_content}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                vision_inputs, video_inputs = process_vision_info(messages)
                return key, txt, views, text, vision_inputs, video_inputs

            with ThreadPoolExecutor(max_workers=24) as executor:
                results = list(executor.map(preprocess_message, miss_items))

            # === inference-only ===
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for key, txt, views, text, vision_inputs, video_inputs in results:
                    cache_path = self._cache_path(key, txt, views)
                    if cache_path.exists():
                        pooled = torch.load(cache_path, map_location="cpu", weights_only=True)
                        pooled = pooled.pin_memory().to(device=device, non_blocking=True, dtype=torch.bfloat16)
                        pooled_vl_tokens_dict[key] = pooled
                        continue

                    inputs = self.processor(
                        text=[text],
                        images=vision_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

                    outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True)
                    vl_tokens = outputs.hidden_states[-1]
                    pooled = vl_tokens.mean(dim=1, keepdim=True)

                    self._atomic_save(pooled.detach().to("cpu", dtype=torch.float16), cache_path)
                    self._enforce_cache_limit(max_gb=20)
                    pooled_vl_tokens_dict[key] = pooled.to(dtype=torch.bfloat16)

        # --- 3️⃣ 순서 복원 ---
        pooled_vl_tokens = [pooled_vl_tokens_dict[k] for k in cache_keys if k in pooled_vl_tokens_dict]
        vl_tokens = torch.cat(pooled_vl_tokens, dim=0)

        # --- 4️⃣ Action 예측 ---
        z_chunk = z_chunk.to(device=device, dtype=vl_tokens.dtype)
        pred_actions, delta = self.action_expert(vl_tokens, z_chunk)
        return pred_actions, delta
    
class Not_freeze_QwenVLAForAction(nn.Module):
    def __init__(self,
                 vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 action_dim=7, horizon=8, hidden_dim=1024,
                 cache_dir="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
                 finetune_vl="none",  # "none" | "lora" | "full"
                 lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 unfreeze_last_n=2):
        super().__init__()
        print(f"🚀 Loading Qwen-VL backbone: {vl_model_name}")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.finetune_vl = finetune_vl
        self.cache_mode = "on"  # lazy cache 모드 사용

        # =============================
        # 1️⃣ Qwen-VL backbone 로딩
        # =============================
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.processor = AutoProcessor.from_pretrained(vl_model_name)
        self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vl_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": local_rank},
            low_cpu_mem_usage=True,
        )

        # =============================
        # 2️⃣ Action Expert
        # =============================
        self.action_expert = QwenActionExpert(
            vl_dim=self.vl_model.config.hidden_size,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim
        ).to(dtype=torch.bfloat16)

        # =============================
        # 3️⃣ Fine-tuning 설정
        # =============================
        for p in self.vl_model.parameters():
            p.requires_grad = False

        if finetune_vl == "lora":
            print("💡 Applying LoRA fine-tuning...")
            self._inject_lora_to_vl(lora_r, lora_alpha, lora_dropout)
        elif finetune_vl == "full":
            print(f"💡 Unfreezing last {unfreeze_last_n} layers...")
            self._selective_unfreeze_vl(unfreeze_last_n)
        else:
            print("🧊 Using frozen VL backbone (feature-only).")
            
        if finetune_vl in ["lora", "full"]:
            self.cache_mode = "off"   # LoRA 학습 중에는 항상 end-to-end로 인코딩
        else:
            self.cache_mode = "on"    # Frozen일 때만 lazy cache 허용

    # =============================
    # LoRA 및 selective unfreeze 부분 동일
    # =============================
    def _inject_lora_to_vl(self, r, alpha, dropout):
        from peft import LoraConfig, get_peft_model
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
        )
        self.vl_model = get_peft_model(self.vl_model, cfg)
        for n, p in self.vl_model.named_parameters():
            p.requires_grad = "lora" in n

    def _selective_unfreeze_vl(self, last_n=2):
        blocks = None
        for attr in ["model.layers", "transformer.blocks", "layers"]:
            try:
                blocks = eval(f"self.vl_model.{attr}")
                break
            except Exception:
                pass
        if blocks is None:
            for p in self.vl_model.parameters():
                p.requires_grad = True
            return
        for i, blk in enumerate(blocks):
            trainable = i >= (len(blocks) - last_n)
            for p in blk.parameters():
                p.requires_grad = trainable

    # =============================
    # 캐시 관련 유틸
    # =============================
    def _cache_path(self, key: str, txt: str, views: list[str | None]) -> Path:
        vlist = [v for v in views if v is not None]
        raw = key + "||" + txt + "||" + "|".join(vlist)
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.pt"

    @staticmethod
    def _atomic_save(tensor_cpu: torch.Tensor, path: Path):
        tmp = path.with_suffix(".pt.tmp")
        with open(str(path) + ".lock", "w") as lockfile:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            if not path.exists():
                torch.save(tensor_cpu, tmp)
                os.replace(tmp, path)
            fcntl.flock(lockfile, fcntl.LOCK_UN)

    def _enforce_cache_limit(self, max_gb=20):
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt"))
        if total_bytes > max_gb * (1024 ** 3):
            all_files = sorted(self.cache_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
            while total_bytes > max_gb * (1024 ** 3) and all_files:
                f = all_files.pop(0)
                total_bytes -= f.stat().st_size
                f.unlink(missing_ok=True)
            print(f"⚠️ Cache trimmed to {max_gb}GB.")

    # =============================
    # 통합 인코딩 (lazy cache fallback)
    # =============================
    def _encode_lazy_cache(self, text_inputs, image_inputs, cache_keys, device):
        pooled_vl_tokens_dict = {}
        miss_items = []

        # --- 캐시 히트 체크
        for txt, views, key in zip(text_inputs, image_inputs, cache_keys):
            path = self._cache_path(key, txt, views)
            if path.exists():
                pooled = torch.load(path, map_location="cpu")
                pooled_vl_tokens_dict[key] = pooled.to(device, dtype=torch.bfloat16)
            else:
                miss_items.append((txt, views, key))

        # --- 캐시 미스 → 즉시 VL 인코딩 수행
        if miss_items:
            def preprocess(args):
                txt, views, key = args
                msg_content = [{"type": "image", "image": v} for v in views if v is not None]
                msg_content.append({"type": "text", "text": txt})
                messages = [{"role": "user", "content": msg_content}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                vision_inputs, _ = process_vision_info(messages)
                return key, txt, views, text, vision_inputs

            with ThreadPoolExecutor(max_workers=4) as ex:
                results = list(ex.map(preprocess, miss_items))

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for key, txt, views, text, vision_inputs in results:
                    path = self._cache_path(key, txt, views)
                    inputs = self.processor(
                        text=[text], images=vision_inputs,
                        padding=True, return_tensors="pt"
                    ).to(device=device, dtype=torch.bfloat16)

                    outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True)
                    vl_tokens = outputs.hidden_states[-1]
                    pooled = vl_tokens.mean(dim=1, keepdim=True)

                    # 캐시 저장
                    self._atomic_save(pooled.detach().to("cpu", dtype=torch.float16), path)
                    self._enforce_cache_limit(max_gb=20)
                    pooled_vl_tokens_dict[key] = pooled

        # --- 순서 복원
        ordered = [pooled_vl_tokens_dict[k] for k in cache_keys if k in pooled_vl_tokens_dict]
        vl_tokens = torch.cat(ordered, dim=0)
        return vl_tokens

    # =============================
    # Forward (통합 구조)
    # =============================
    def forward(self, text_inputs, image_inputs, z_chunk, cache_keys=None):
        device = next(self.parameters()).device
        if cache_keys is None:
            cache_keys = [f"idx={i}" for i in range(len(text_inputs))]

        # 🔹 캐시 사용 여부 분기
        if getattr(self, "cache_mode", "on") == "off":
            # LoRA or Full Fine-tuning → 항상 새로 인코딩 (cache 비활성화)
            msg_batch = []
            for txt, views in zip(text_inputs, image_inputs):
                msg_content = [{"type": "image", "image": v} for v in views if v is not None]
                msg_content.append({"type": "text", "text": txt})
                msg_batch.append({"role": "user", "content": msg_content})

            text = self.processor.apply_chat_template(msg_batch, tokenize=False, add_generation_prompt=False)
            vision_inputs, video_inputs = process_vision_info(msg_batch)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                inputs = self.processor(
                    text=[text],
                    images=vision_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(device=device, dtype=torch.bfloat16)

                outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True)
                vl_tokens = outputs.hidden_states[-1]
                pooled_vl_tokens = vl_tokens.mean(dim=1, keepdim=True)

        else:
            # Frozen 모드일 때만 lazy cache 사용
            pooled_vl_tokens = self._encode_lazy_cache(text_inputs, image_inputs, cache_keys, device)

        # 🔹 Action Expert
        z_chunk = z_chunk.to(device=device, dtype=pooled_vl_tokens.dtype)
        pred_actions, delta = self.action_expert(pooled_vl_tokens, z_chunk)

        # 🔹 메모리 해제
        del pooled_vl_tokens
        torch.cuda.empty_cache()
        return pred_actions, delta

