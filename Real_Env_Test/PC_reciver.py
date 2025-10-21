import os, time, json, cv2, zmq, numpy as np
from collections import defaultdict
from datetime import datetime

# ==============================
# 세션 폴더 생성 (시간 기반)
# ==============================
session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = f"./recv_all_{session_time}"
os.makedirs(BASE_DIR, exist_ok=True)

# View별 폴더 생성
for view in range(1, 6):
    view_dir = os.path.join(BASE_DIR, f"View{view}")
    if view <= 4:
        os.makedirs(os.path.join(view_dir, "left"), exist_ok=True)
        os.makedirs(os.path.join(view_dir, "right"), exist_ok=True)
    else:
        os.makedirs(view_dir, exist_ok=True)

print(f"📁 Save directory: {BASE_DIR}")

# ==============================
# ZMQ 수신 설정
# ==============================
ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PULL)
sock.setsockopt(zmq.RCVHWM, 5000)
sock.bind("tcp://0.0.0.0:5555")
print("✅ Listening on tcp://0.0.0.0:5555")

cnt = defaultdict(int)
t0 = time.time()

# ==============================
# 수신 루프
# ==============================
while True:
    parts = sock.recv_multipart()
    if len(parts) < 2:
        continue

    meta = json.loads(parts[0].decode("utf-8"))
    jpg = parts[1]
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

    cam = meta.get("camera", "unknown")  # 예: ZED1_LEFT, ZED3_RIGHT, OAK
    ts = meta.get("timestamp", time.time())
    cnt[cam] += 1

    # ==============================
    # View 및 폴더 결정
    # ==============================
    cam_upper = cam.upper()
    if "ZED1" in cam_upper:
        view_dir = os.path.join(BASE_DIR, "View1")
    elif "ZED2" in cam_upper:
        view_dir = os.path.join(BASE_DIR, "View2")
    elif "ZED3" in cam_upper:
        view_dir = os.path.join(BASE_DIR, "View3")
    elif "ZED4" in cam_upper:
        view_dir = os.path.join(BASE_DIR, "View4")
    elif "OAK" in cam_upper:
        view_dir = os.path.join(BASE_DIR, "View5")
    else:
        view_dir = os.path.join(BASE_DIR, "Unknown")

    # left/right 하위 폴더 지정
    if "LEFT" in cam_upper:
        save_dir = os.path.join(view_dir, "left")
    elif "RIGHT" in cam_upper:
        save_dir = os.path.join(view_dir, "right")
    else:
        save_dir = view_dir

    os.makedirs(save_dir, exist_ok=True)

    # ==============================
    # 이미지 저장 (파일명 기존 방식 유지)
    # ==============================
    filename = f"{cam}_{ts:.3f}.jpg"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, img)

    # ==============================
    # 주기적 상태 출력
    # ==============================
    if time.time() - t0 > 2:
        elapsed = time.time() - t0
        line = " | ".join([f"{k}:{cnt[k]/elapsed:.1f}fps" for k in sorted(cnt)])
        print(line)
        cnt = defaultdict(int)
        t0 = time.time()
