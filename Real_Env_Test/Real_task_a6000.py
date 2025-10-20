# ===========================================
# Real_task_final_jetson_opt.py (Jetson 최적화 + 안정성 강화)
# ===========================================
# pip install xformers
# pip install pillow
# pip install --upgrade transformers

import sys
import time
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append("/home/najo/NAS/VLA/Qwen2.5-VL-3B-_OCT_FPI_Action_Model")

from model import QwenVLAForAction
from Total_Dataset import BridgeRawSequenceDataset, collate_fn  # (호환성 유지용)

# ===========================================
# 1️⃣ 기본 설정
# ===========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "../checkpoints/qwen_vla_final_1000.pt"
VL_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

torch.backends.cuda.matmul.allow_tf32 = True  # Ampere 이상 TF32 허용
torch.backends.cudnn.benchmark = True         # 입력 크기 고정 시 Convolution 최적화
torch.set_float32_matmul_precision("high")    # GEMM 고정 정밀도 상향

# ===========================================
# 2️⃣ 모델 로드
# ===========================================
print("🚀 Loading model...")
model = QwenVLAForAction(
    vl_model_name=VL_MODEL_NAME,
    action_dim=7,
    horizon=8,
    hidden_dim=1024,
    cache_dir="./cache/qwen_vl_features"
).to(DEVICE)

model.set_cache(False)  # 캐시 비활성화 (추론 단계에서는 비추천)
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()
print("✅ Model loaded and frozen.\n")

# ===========================================
# 3️⃣ Processor 및 입력 데이터 준비
# ===========================================
processor = AutoProcessor.from_pretrained(VL_MODEL_NAME)

# 테스트용 이미지
image_path = "/home/najo/NAS/VLA/dataset/part1/ZED_Captures_2th/view1/left/zed_41182735_left_1759119936.930.jpg"
image = Image.open(image_path).convert("RGB")

instruction = "Move the gripper towards the white block."
z_chunk = torch.zeros((1, 8, 7), dtype=torch.bfloat16, device=DEVICE)

# processor는 tokenizer 용도로만 사용 (이미지는 그대로 모델에 전달)
_ = processor(text=instruction, images=image, return_tensors="pt").to(DEVICE)

# ===========================================
# 4️⃣ Warm-up 단계 (JIT, CUDA 커널 초기화)
# ===========================================
print("🔥 Warming up model (3 iters)...")
torch.cuda.synchronize()
with torch.no_grad():
    for _ in range(3):
        _ = model(
            text_inputs=instruction,
            image_inputs=[[image]],
            z_chunk=z_chunk
        )
torch.cuda.synchronize()
print("✅ Warm-up complete.\n")

# ===========================================
# 5️⃣ 정밀한 GPU Inference 시간 측정
# ===========================================
print("⚡ Measuring precise GPU inference time...")

num_runs = 10
times = []

with torch.no_grad():
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = model(
            text_inputs=instruction,
            image_inputs=[[image]],
            z_chunk=z_chunk
        )
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)

mean_time = np.mean(times)
std_time = np.std(times)
print(f"✅ Mean Inference Time: {mean_time:.2f} ms ± {std_time:.2f} ms\n")

# ===========================================
# 6️⃣ 단일 추론 결과 확인
# ===========================================
print("🎯 Running final single inference...")

with torch.no_grad():
    output = model(
        text_inputs=instruction,
        image_inputs=[[image]],
        z_chunk=z_chunk
    )

if isinstance(output, (tuple, list)):
    pred_action = output[0]
else:
    pred_action = output

print("Predicted action:", pred_action)
print("Shape:", pred_action.shape)
print("✅ Completed successfully.")
