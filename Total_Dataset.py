import json
import os
import re
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tqdm import tqdm

def collate_fn(batch):
    images = [b["images"] for b in batch]
    instructions = [b["instruction"] for b in batch]
    confidences = [b["confidence"] for b in batch]
    cache_keys = [b["cache_key"] for b in batch]

    actions = torch.stack([b["actions"] for b in batch], dim=0)  # [B, H, 7]
    return {
        "images": images,
        "instruction": instructions,
        "actions": actions,
        "confidence": confidences,
        "cache_keys": cache_keys,
    }

class insertionMeca500Dataset(Dataset):
    """
    Meca500 로봇의 멀티뷰 이미지와 로봇 상태를 JSON 파일로부터 로드합니다.
    - Action: 연속된 ee_pose 간의 차이(delta)에 마지막 차원 1을 추가하여 7D로 만듭니다.
    - Observation: Left 카메라와 OAK 카메라 이미지만 사용합니다.
    """
    def __init__(self, json_path, horizon=8, instruction="Approach the white square silicone"):
        self.json_path = json_path
        self.horizon = horizon
        self.instruction = instruction
        
        with open(self.json_path, 'r') as f:
            self.trajectory_data = json.load(f)

        if len(self.trajectory_data) < 2:
            raise ValueError(f"Dataset {json_path} must have at least 2 timesteps to calculate delta actions.")

        absolute_poses = np.array([
            item['robot_state']['ee_pose'] for item in self.trajectory_data
        ], dtype=np.float32)
        
        # --- ✅ [수정된 부분] Action을 7D로 확장 ---
        # 1. 기존과 같이 6D delta pose를 계산합니다.
        delta_poses_6d = absolute_poses[1:] - absolute_poses[:-1]
        
        # 2. delta pose 개수만큼 1로 채워진 새로운 열을 만듭니다.
        num_actions = delta_poses_6d.shape[0]
        last_dim_ones = np.ones((num_actions, 1), dtype=np.float32)
        
        # 3. 두 배열을 수평으로 결합하여 (N, 7) 형태의 최종 action을 만듭니다.
        self.actions = np.concatenate([delta_poses_6d, last_dim_ones], axis=1)
        # ---------------------------------------------
        
        first_img_keys = sorted(self.trajectory_data[0]['images'].keys())
        self.view_keys = [k for k in first_img_keys if k.endswith('_left') or k.endswith('_oak')]
        self.num_views = len(self.view_keys)
        self.samples = self._index_chunks()

    def _index_chunks(self):
        num_actions = len(self.actions)
        chunk_count = max(num_actions - self.horizon + 1, 0)
        return list(range(chunk_count))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx = self.samples[idx]
        t = start_idx

        # === Image Loading ===
        current_data_point = self.trajectory_data[t]
        image_paths_dict = current_data_point['images']
        views = [f"file://{image_paths_dict[key]}" for key in self.view_keys]

        # === Action sequence Loading (Bridge와 동일한 로직) ===
        start = start_idx
        end = start_idx + self.horizon
        
        # ✅ Bridge의 '마지막 action 반복' 패딩 로직 추가
        if end > len(self.actions):
            act_seq = self.actions[start:len(self.actions)]
            last_action = act_seq[-1] if len(act_seq) > 0 else np.zeros(self.actions.shape[1], dtype=np.float32)
            repeat_len = end - len(self.actions)
            pad = np.tile(last_action, (repeat_len, 1))
            act_seq = np.concatenate([act_seq, pad], axis=0)
        else:
            act_seq = self.actions[start:end]
            
        act_seq = torch.tensor(act_seq, dtype=torch.float32)

        # === Language & Other Info ===
        lang = self.instruction
        confidence = 1.0
        cache_key = f"{os.path.basename(self.json_path)}::t={t}"

        return {
            "images": views, 
            "actions": act_seq, 
            "instruction": lang, 
            "confidence": confidence, 
            "cache_key": cache_key
        }
        
def infer_lang_from_path(traj_path):
        match = re.search(r"datacol2_([^/]+)/([^/]+)/", traj_path)
        if match:
            env = match.group(1).replace("_", " ")
            task = match.group(2).replace("_", " ")
            return f"{task} task in the {env}"
        else:
            return "<no_lang>"

class BridgeRawSequenceDataset(Dataset):
    def __init__(self, root, horizon=8, max_traj=None):
        self.root = root
        self.horizon = horizon

        policy_files1 = glob.glob(
            os.path.join(
                root, "bridge_data_v2", "datacol2_*", "*", "*", "*", "raw", "traj_group*", "traj*", "policy_out.pkl"
            ),
            recursive=False
        )
        policy_files2 = glob.glob(
            os.path.join(
                root, "bridge_data_v1", "berkeley", "*", "*", "*", "raw", "traj_group*", "traj*", "policy_out.pkl"
            ),
            recursive=False
        )
        
        policy_files3 = glob.glob(
            os.path.join(
                root, "bridge_data_v2", "deepthought_*", "*", "*", "*", "raw", "traj_group*", "traj*", "policy_out.pkl"
            ),
            recursive=False
        )
        
        all_policy_files = policy_files2 + policy_files3
        self.traj_paths = [os.path.dirname(p) for p in all_policy_files]
        print(f"✅ Found {len(self.traj_paths)} trajectories across all tasks")
        
        if max_traj:
            self.traj_paths = self.traj_paths[:max_traj]

        # === 전체 dataset에서 최대 view 수 계산 ===
        self.max_views = 0
        for traj_path in tqdm(self.traj_paths, desc="Scanning max views"):
            view_dirs = [d for d in os.listdir(traj_path) if d.startswith("images")]
            self.max_views = max(self.max_views, len(view_dirs))
        print(f"✅ Max number of views across dataset: {self.max_views}")

        # === chunk 단위 인덱싱 ===
        self.samples = self._index_chunks()

    def _index_chunks(self):
        samples = []
        for traj_path in tqdm(self.traj_paths, desc="Indexing Chunks"):
            img_dir = os.path.join(traj_path, "images0")
            imgs = sorted(glob.glob(os.path.join(img_dir, "im_*.jpg")))
            if not imgs:
                continue
            T = len(imgs)
            chunk_count = max(T - self.horizon + 1, 0)
            for i in range(chunk_count):
                samples.append((traj_path, i))
        print(f"✅ Indexed {len(samples)} chunks from {len(self.traj_paths)} trajectories")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_path, start_idx = self.samples[idx]

        # === 현재 t 시점의 view 4장만 불러오기 ===
        view_dirs = sorted([d for d in os.listdir(traj_path) if d.startswith("images")])
        num_views = len(view_dirs)

        t = start_idx  # 현재 시점
        views = []
        for v in range(num_views):
            img_path = os.path.join(traj_path, f"images{v}", f"im_{t}.jpg")
            if os.path.exists(img_path):
                views.append(f"file://{os.path.abspath(img_path)}")

        # 패딩 (view 부족 시)
        if len(views) < self.max_views:
            views += [None] * (self.max_views - len(views))
        views = views[:self.max_views]

        # === Action sequence ===
        with open(os.path.join(traj_path, "policy_out.pkl"), "rb") as f:
            actions = pickle.load(f)
            if isinstance(actions[0], dict):
                actions = [a.get("actions") for a in actions if "actions" in a]
        actions = np.array(actions, dtype=np.float32)

        # ✅ 부족 시 horizon을 "마지막 action 반복"으로 채우기
        start = start_idx
        end = start_idx + self.horizon
        if end > len(actions):
            act_seq = actions[start:len(actions)]
            last_action = act_seq[-1] if len(act_seq) > 0 else np.zeros(actions.shape[1], dtype=np.float32)
            repeat_len = end - len(actions)
            pad = np.tile(last_action, (repeat_len, 1))
            act_seq = np.concatenate([act_seq, pad], axis=0)
        else:
            act_seq = actions[start:end]

        act_seq = torch.tensor(act_seq, dtype=torch.float32)  # shape: [horizon, 7]

        # === Language ===
        lang_path = os.path.join(traj_path, "lang.txt")
        lang = ""
        confidence = 0.5
        if os.path.exists(lang_path):
            with open(lang_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(lines) == 1:
                lang = lines[0]
            elif len(lines) >= 2:
                lang = lines[0]
                for line in lines[1:]:
                    if "confidence" in line.lower():
                        try:
                            confidence = float(line.split(":")[-1].strip())
                        except:
                            confidence = 0.5
        else:
            lang = infer_lang_from_path(traj_path)
        
        cache_key = f"{traj_path}::t={t}"

        return {"images": views, "actions": act_seq, "instruction": lang, "confidence": confidence, "cache_key": cache_key}
