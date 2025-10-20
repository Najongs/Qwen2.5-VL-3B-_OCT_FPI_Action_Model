import torch
import matplotlib.pyplot as plt
import numpy as np
from model import QwenVLAForAction
from Total_Dataset import BridgeRawSequenceDataset, collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "./checkpoints/qwen_vla_final_1000.pt"
VL_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

model = QwenVLAForAction(
    vl_model_name=VL_MODEL_NAME,
    action_dim=7,
    horizon=8,
    hidden_dim=1024,
    cache_dir="./cache/qwen_vl_features"
).to(DEVICE)

model.set_cache(False)  # 캐시 기능 자체는 비활성

checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()