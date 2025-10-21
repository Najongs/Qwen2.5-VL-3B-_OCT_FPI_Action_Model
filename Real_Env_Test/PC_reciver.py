import os, time, json, cv2, zmq, numpy as np
import threading
from queue import Queue
from collections import defaultdict
from datetime import datetime

# ==============================
# 1️⃣ 세션 폴더 생성
# ==============================
session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = f"./recv_all_{session_time}"
os.makedirs(BASE_DIR, exist_ok=True)

for view in range(1, 6):
    view_dir = os.path.join(BASE_DIR, f"View{view}")
    if view <= 4:
        os.makedirs(os.path.join(view_dir, "left"), exist_ok=True)
        os.makedirs(os.path.join(view_dir, "right"), exist_ok=True)
    else:
        os.makedirs(view_dir, exist_ok=True)

print(f"📁 Save directory: {BASE_DIR}")

# ==============================
# 2️⃣ 설정 및 파라미터
# ==============================
ZED_SERIAL_TO_VIEW = {
    "41182735": "View1",
    "49429257": "View2",
    "44377151": "View3",
    "49045152": "View4",
}
OAK_KEYWORD = "OAK"

# 모니터링 임계값
STALL_SEC = 3.0
WARN_NET_MS = 150.0
WARN_TOTAL_MS = 1000.0
STATUS_PERIOD = 1.0
FPS_PERIOD = 2.0

# ==============================
# 3️⃣ 비동기 이미지 저장
# ==============================
class AsyncImageWriter(threading.Thread):
    def __init__(self, max_queue=5000):
        super().__init__(daemon=True)
        self.q = Queue(max_queue)
        self.stop_flag = threading.Event()

    def submit(self, path, img):
        if not self.stop_flag.is_set():
            try:
                self.q.put_nowait((path, img))
            except:
                print(f"[Writer] Queue full, dropping frame: {path}")

    def run(self):
        while True:
            try:
                path, img = self.q.get(timeout=0.1)
            except:
                if self.stop_flag.is_set() and self.q.empty():
                    break
                continue
            try:
                cv2.imwrite(path, img)
            except Exception as e:
                print(f"[Writer] Error saving {path}: {e}")

    def stop(self):
        print(f"🕒 Flushing remaining {self.q.qsize()} images before shutdown...")
        self.stop_flag.set()
        while not self.q.empty():
            time.sleep(0.1)
        print("🛑 Writer thread stopped.")

# ==============================
# 4️⃣ ZMQ 수신 설정
# ==============================
ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PULL)
sock.setsockopt(zmq.RCVHWM, 5000)
sock.setsockopt(zmq.RCVBUF, 4 * 1024 * 1024)  # ✅ 버퍼 확장 (4MB)
sock.bind("tcp://0.0.0.0:5555")
print("✅ Listening on tcp://0.0.0.0:5555")

writer = AsyncImageWriter()
writer.start()

# ==============================
# 5️⃣ 상태 변수
# ==============================
cnt = defaultdict(int)
fail_count = defaultdict(int)
last_ts = {}
delta_dict = {}
latest_recv = {}
last_recv_wall = {}
cam_save_dir_cache = {}

t0 = time.time()
last_status_print = time.time()
last_fps_print = time.time()

# ==============================
# 6️⃣ 유틸: 카메라별 디렉토리 캐시
# ==============================
def get_save_dir_for_cam(cam_name: str):
    if cam_name in cam_save_dir_cache:
        return cam_save_dir_cache[cam_name]
    cam_lower = cam_name.lower()
    view_dir = os.path.join(BASE_DIR, "Unknown")

    if "zed" in cam_lower:
        for serial, view_name in ZED_SERIAL_TO_VIEW.items():
            if serial in cam_lower:
                view_dir = os.path.join(BASE_DIR, view_name)
                break
    elif OAK_KEYWORD.lower() in cam_lower:
        view_dir = os.path.join(BASE_DIR, "View5")

    if "left" in cam_lower:
        save_dir = os.path.join(view_dir, "left")
    elif "right" in cam_lower:
        save_dir = os.path.join(view_dir, "right")
    else:
        save_dir = view_dir

    os.makedirs(save_dir, exist_ok=True)
    cam_save_dir_cache[cam_name] = save_dir
    return save_dir

# ==============================
# 7️⃣ 수신 루프
# ==============================
try:
    while True:
        try:
            parts = sock.recv_multipart()
        except Exception as e:
            print(f"[WARN] recv_multipart error: {e}")
            continue

        # ✅ multipart 무결성 검사
        if len(parts) < 2:
            print("[WARN] multipart length < 2, skip")
            continue

        meta_raw, jpg = parts[0], parts[1]
        if not jpg or len(jpg) < 5000:
            print(f"[WARN] Incomplete JPEG received (len={len(jpg)})")
            continue

        try:
            meta = json.loads(meta_raw.decode("utf-8"))
        except Exception as e:
            print(f"[WARN] meta decode error: {e}")
            continue

        cam = meta.get("camera", "unknown")
        ts = float(meta.get("timestamp", 0.0))
        send_time = float(meta.get("send_time", 0.0))
        recv_time = time.time()

        # ✨✨✨ 핵심 수정 부분: 핸드셰이크 패킷 (ts=0.0) 건너뛰기 ✨✨✨
        if ts == 0.0:
             print(f"⚪️ Received non-data message (ts=0.0) from {cam}, skipping.")
             continue # 다음 메시지 수신으로 넘어감
        # ✨✨✨ 수정 완료 ✨✨✨

        net_delay = (recv_time - send_time) if send_time > 0 else 0.0
        total_delay = (recv_time - ts) if ts > 0 else 0.0

        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            fail_count[cam] += 1
            print(f"[WARN] JPEG decode failed for {cam} (fail #{fail_count[cam]})")
            continue

        cnt[cam] += 1
        latest_recv[cam] = {
            "capture_time": ts,
            "send_time": send_time,
            "recv_time": recv_time,
            "net_delay": net_delay,
            "total_delay": total_delay
        }
        last_recv_wall[cam] = recv_time

        # Δt 계산 (촬영 시각 기반)
        if cam in last_ts and ts > 0:
            delta = ts - last_ts[cam]
            if delta > 0:
                delta_dict[cam] = delta
        if ts > 0:
            last_ts[cam] = ts

        # 이미지 저장
        save_dir = get_save_dir_for_cam(cam)
        filename = f"{cam}_{ts:.3f}.jpg" if ts > 0 else f"{cam}_{recv_time:.3f}.jpg"
        writer.submit(os.path.join(save_dir, filename), img)

        # ==========================
        # 실시간 상태 모니터링
        # ==========================
        now = time.time()
        if now - last_status_print >= STATUS_PERIOD:
            print("\n📡 [Real-Time Status]")
            for k in sorted(latest_recv.keys()):
                data = latest_recv[k]
                d = delta_dict.get(k, 0.0)
                fps = (1.0 / d) if d > 0 else 0.0

                net_ms = data['net_delay'] * 1000.0
                total_ms = data['total_delay'] * 1000.0
                net_flag = "⚠️" if net_ms > WARN_NET_MS else "  "
                total_flag = "⚠️" if total_ms > WARN_TOTAL_MS else "  "

                stall_mark = ""
                last_wall = last_recv_wall.get(k, 0.0)
                if (now - last_wall) >= STALL_SEC:
                    stall_mark = " ⛔STALLED"

                fail_mark = f" (fail:{fail_count.get(k,0)})" if fail_count.get(k,0) > 0 else ""

                print(
                    f"  {k:<25} | "
                    f"cap:{data['capture_time']:.3f} | "
                    f"send:{data['send_time']:.3f} | "
                    f"recv:{data['recv_time']:.3f} | "
                    f"Δnet:{net_ms:6.1f}ms{net_flag} | "
                    f"Δtotal:{total_ms:6.1f}ms{total_flag} | "
                    f"{fps:5.2f}fps{stall_mark}{fail_mark}"
                )
            last_status_print = now

        # ==========================
        # 평균 FPS 출력
        # ==========================
        if now - last_fps_print >= FPS_PERIOD:
            elapsed = now - last_fps_print
            if elapsed > 0: # Prevent division by zero if loop runs extremely fast
                line = " | ".join([f"{k}:{cnt[k]/elapsed:.1f}fps" for k in sorted(cnt)])
                print("📊 평균:", line)
            cnt = defaultdict(int)
            last_fps_print = now

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

finally:
    print("\n🧾 Final per-camera summary:")
    for k in sorted(latest_recv.keys()):
        data = latest_recv[k]
        print(
            f"  {k:<25} | "
            f"cap_last:{data['capture_time']:.3f} | "
            f"send_last:{data['send_time']:.3f} | "
            f"recv_last:{data['recv_time']:.3f} | "
            f"fail:{fail_count.get(k,0)}"
        )
    writer.stop()
    writer.join()
    print("✅ Receiver shutdown complete.")