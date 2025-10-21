import os, time, json, cv2, zmq, numpy as np
import threading
from queue import Queue, Empty
from collections import defaultdict
from datetime import datetime
import struct # <-- 로봇 데이터 언패킹 위해 추가
import csv # <-- CSV 저장을 위해 추가

# ==============================
# 1️⃣ 세션 폴더 생성
# ==============================
session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = f"./recv_all_{session_time}"
os.makedirs(BASE_DIR, exist_ok=True)
# 카메라 뷰 폴더 생성
for view in range(1, 6):
    view_dir = os.path.join(BASE_DIR, f"View{view}")
    if view <= 4: # ZED
        os.makedirs(os.path.join(view_dir, "left"), exist_ok=True)
        os.makedirs(os.path.join(view_dir, "right"), exist_ok=True)
    else: # OAK
        os.makedirs(view_dir, exist_ok=True)
print(f"📁 Save directory: {BASE_DIR}")
# 로봇 데이터 저장 파일 경로 (최종 저장용)
ROBOT_CSV_PATH = os.path.join(BASE_DIR, f"robot_state_{session_time}.csv")

# ==============================
# 2️⃣ 설정 및 파라미터
# ==============================
ZED_SERIAL_TO_VIEW = {
    "41182735": "View1", "49429257": "View2", "44377151": "View3", "49045152": "View4",
}
OAK_KEYWORD = "OAK"

# 모니터링 임계값
STALL_SEC = 3.0
WARN_NET_MS = 150.0
WARN_TOTAL_MS = 1000.0
STATUS_PERIOD = 1.0
FPS_PERIOD = 2.0

# 로봇 ZMQ 설정
ZMQ_ROBOT_PUB_ADDRESS = "10.130.41.111" # 송신측 IP
ZMQ_ROBOT_PUB_PORT = 5556
ZMQ_ROBOT_TOPIC = b"robot_state"
ROBOT_PAYLOAD_FORMAT = '<ddf12f'
ROBOT_PAYLOAD_SIZE = struct.calcsize(ROBOT_PAYLOAD_FORMAT) # 68 bytes

# 카메라 ZMQ 설정
ZMQ_CAM_PULL_PORT = 5555

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
            # Use task_done for proper queue joining
            try: self.q.put_nowait((path, img))
            except: print(f"[Writer] Queue full, dropping frame: {path}")

    def run(self):
        while True:
            try:
                path, img = self.q.get(timeout=0.1)
                try:
                    cv2.imwrite(path, img)
                except Exception as e:
                    print(f"[Writer] Error saving {path}: {e}")
                finally:
                    self.q.task_done() # Indicate task completion
            except Empty:
                if self.stop_flag.is_set() and self.q.empty():
                    break
                continue

    def stop(self):
        print(f"🕒 Flushing remaining {self.q.qsize()} images before shutdown...")
        self.stop_flag.set()
        self.q.join() # Wait for all tasks to be marked done
        print("🛑 Writer thread stopped.")

# ==============================
# 4️⃣ ZMQ 수신 설정
# ==============================
ctx = zmq.Context.instance()
cam_sock = ctx.socket(zmq.PULL)
cam_sock.setsockopt(zmq.RCVHWM, 5000)
cam_sock.setsockopt(zmq.RCVBUF, 4 * 1024 * 1024)
cam_bind_addr = f"tcp://0.0.0.0:{ZMQ_CAM_PULL_PORT}"
cam_sock.bind(cam_bind_addr)
print(f"✅ Camera PULL listening on {cam_bind_addr}")

robot_sock = ctx.socket(zmq.SUB)
robot_sock.setsockopt(zmq.RCVHWM, 100)
robot_connect_addr = f"tcp://{ZMQ_ROBOT_PUB_ADDRESS}:{ZMQ_ROBOT_PUB_PORT}"
robot_sock.connect(robot_connect_addr)
robot_sock.subscribe(ZMQ_ROBOT_TOPIC)
print(f"✅ Robot SUB connected to {robot_connect_addr} (Topic: '{ZMQ_ROBOT_TOPIC.decode()}')")

poller = zmq.Poller()
poller.register(cam_sock, zmq.POLLIN)
poller.register(robot_sock, zmq.POLLIN)

writer = AsyncImageWriter()
writer.start()

# ==============================
# 5️⃣ 상태 변수
# ==============================
cam_cnt = defaultdict(int)
cam_fail_count = defaultdict(int)
cam_last_ts = {}
cam_delta_dict = {}
cam_latest_recv = {}
cam_last_recv_wall = {}
cam_save_dir_cache = {}

robot_cnt = 0
robot_latest_state = {}
robot_last_recv_wall = 0.0
robot_data_buffer = [] # CSV 저장을 위한 **메모리** 버퍼 (종료 시 저장)

t0 = time.time()
last_status_print = time.time()
last_fps_print = time.time()

# ==============================
# 6️⃣ 유틸: 카메라/로봇 저장 함수
# ==============================
def get_save_dir_for_cam(cam_name: str):
    if cam_name in cam_save_dir_cache: return cam_save_dir_cache[cam_name]
    cam_lower = cam_name.lower()
    view_dir = os.path.join(BASE_DIR, "Unknown")
    if "zed" in cam_lower:
        for serial, view_name in ZED_SERIAL_TO_VIEW.items():
            if serial in cam_lower: view_dir = os.path.join(BASE_DIR, view_name); break
    elif OAK_KEYWORD.lower() in cam_lower: view_dir = os.path.join(BASE_DIR, "View5")
    if "left" in cam_lower: save_dir = os.path.join(view_dir, "left")
    elif "right" in cam_lower: save_dir = os.path.join(view_dir, "right")
    else: save_dir = view_dir
    os.makedirs(save_dir, exist_ok=True)
    cam_save_dir_cache[cam_name] = save_dir
    return save_dir

# --- [수정] 로봇 데이터 저장 함수 (항상 새로 쓰기 'w' 모드) ---
def save_robot_data_to_csv(data_list, filepath):
    """로봇 상태 데이터를 CSV 파일에 새로 씁니다."""
    if not data_list:
        print("💾 [Robot Save] No robot data to save.")
        return

    # 항상 'w' 모드를 사용하여 파일을 덮어씁니다.
    mode = 'w'
    print(f"💾 Saving {len(data_list)} robot states to {filepath} (Mode: {mode})")
    try:
        with open(filepath, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 항상 헤더 작성
            writer.writerow([
                "recv_timestamp", "origin_timestamp", "send_timestamp", "force_placeholder",
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
                "pose_x", "pose_y", "pose_z", "pose_a", "pose_b", "pose_r"
            ])
            writer.writerows(data_list) # 모든 데이터를 한 번에 쓰기
        print(f"💾✅ Saved robot data successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save robot data to CSV: {e}")


# ==============================
# 7️⃣ 수신 루프 (수정됨 - 주기적 저장 제거)
# ==============================
try:
    while True:
        try:
            socks = dict(poller.poll(timeout=100))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[WARN] poller.poll error: {e}")
            time.sleep(0.1)
            continue

        now_wall = time.time()

        # --- 카메라 소켓 처리 ---
        if cam_sock in socks and socks[cam_sock] == zmq.POLLIN:
            try:
                parts = cam_sock.recv_multipart(zmq.DONTWAIT)
                if len(parts) < 2: print("[WARN] Cam multipart length < 2, skip"); continue
                meta_raw, jpg = parts[0], parts[1]
                if not jpg or len(jpg) < 5000: print(f"[WARN] Incomplete JPEG (len={len(jpg)})"); continue
                try: meta = json.loads(meta_raw.decode("utf-8"))
                except Exception as e: print(f"[WARN] Cam meta decode error: {e}"); continue
                cam = meta.get("camera", "unknown"); ts = float(meta.get("timestamp", 0.0)); send_time = float(meta.get("send_time", 0.0))
                net_delay = (now_wall - send_time) if send_time > 0 else 0.0; total_delay = (now_wall - ts) if ts > 0 else 0.0
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is None: cam_fail_count[cam] += 1; print(f"[WARN] Cam JPEG decode failed for {cam} (fail #{cam_fail_count[cam]})"); continue
                cam_cnt[cam] += 1
                cam_latest_recv[cam] = {"capture_time": ts, "send_time": send_time, "recv_time": now_wall, "net_delay": net_delay, "total_delay": total_delay}
                cam_last_recv_wall[cam] = now_wall
                if cam in cam_last_ts and ts > 0: delta = ts - cam_last_ts[cam];
                if delta > 0: cam_delta_dict[cam] = delta
                if ts > 0: cam_last_ts[cam] = ts
                save_dir = get_save_dir_for_cam(cam); filename = f"{cam}_{ts:.3f}.jpg" if ts > 0 else f"{cam}_{now_wall:.3f}.jpg"
                writer.submit(os.path.join(save_dir, filename), img)
            except zmq.Again: pass
            except Exception as e: print(f"[ERROR] Unhandled camera processing error: {e}")

        # --- 로봇 소켓 처리 ---
        if robot_sock in socks and socks[robot_sock] == zmq.POLLIN:
            try:
                parts = robot_sock.recv_multipart(zmq.DONTWAIT)
                if len(parts) != 2: print(f"[WARN] Robot multipart length != 2 (got {len(parts)}), skip"); continue
                topic, payload = parts[0], parts[1]
                if topic != ZMQ_ROBOT_TOPIC: print(f"[WARN] Unexpected robot topic: {topic}"); continue
                if len(payload) != ROBOT_PAYLOAD_SIZE: print(f"[WARN] Robot payload size mismatch! Expected {ROBOT_PAYLOAD_SIZE}, got {len(payload)}"); continue
                try:
                    unpacked_data = struct.unpack(ROBOT_PAYLOAD_FORMAT, payload)
                    origin_ts, send_ts, force_pl = unpacked_data[0:3]
                    joints, pose = unpacked_data[3:9], unpacked_data[9:15]
                    robot_cnt += 1
                    robot_latest_state = {"recv_time": now_wall, "origin_ts": origin_ts, "send_ts": send_ts, "joints": joints, "pose": pose, "net_delay": now_wall - send_ts, "total_delay": now_wall - origin_ts}
                    robot_last_recv_wall = now_wall
                    robot_data_buffer.append([now_wall] + list(unpacked_data)) # 메모리에 계속 추가
                except struct.error as e: print(f"[WARN] Robot payload unpack error: {e}"); continue
            except zmq.Again: pass
            except Exception as e: print(f"[ERROR] Unhandled robot processing error: {e}")

        # --- 주기적 저장 로직 제거됨 ---

        # --- 실시간 상태 모니터링 ---
        if now_wall - last_status_print >= STATUS_PERIOD:
            print(f"\n--- Status Update ({datetime.now().strftime('%H:%M:%S')}) ---")
            print("📷 Cameras:")
            for k in sorted(cam_latest_recv.keys()): data = cam_latest_recv[k]; d = cam_delta_dict.get(k, 0.0); fps = (1.0 / d) if d > 0 else 0.0; net_ms = data['net_delay'] * 1000.0; total_ms = data['total_delay'] * 1000.0; net_flag = "⚠️" if net_ms > WARN_NET_MS else " "; total_flag = "⚠️" if total_ms > WARN_TOTAL_MS else " "; stall_mark = " ⛔STALLED" if (now_wall - cam_last_recv_wall.get(k, 0.0)) >= STALL_SEC else ""; fail_mark = f" (fail:{cam_fail_count.get(k,0)})" if cam_fail_count.get(k,0) > 0 else ""; print(f"  {k:<25} | cap:{data['capture_time']:.3f} | send:{data['send_time']:.3f} | recv:{data['recv_time']:.3f} | Δnet:{net_ms:6.1f}ms{net_flag} | Δtotal:{total_ms:6.1f}ms{total_flag} | {fps:5.2f}fps{stall_mark}{fail_mark}")
            print("🤖 Robot:")
            if robot_latest_state: r_data = robot_latest_state; r_net_ms = r_data['net_delay'] * 1000.0; r_total_ms = r_data['total_delay'] * 1000.0; r_net_flag = "⚠️" if r_net_ms > WARN_NET_MS else " "; r_total_flag = "❓" if abs(r_total_ms) > WARN_TOTAL_MS else " "; r_stall_mark = " ⛔STALLED" if (now_wall - robot_last_recv_wall) >= STALL_SEC else ""; print(f"  {'Robot State':<25} | ori:{r_data['origin_ts']:.3f} | send:{r_data['send_ts']:.3f} | recv:{r_data['recv_time']:.3f} | Δnet:{r_net_ms:6.1f}ms{r_net_flag} | Δtotal:{r_total_ms:6.1f}ms{r_total_flag} | J1:{r_data['joints'][0]:<6.1f} Px:{r_data['pose'][0]:<7.1f}{r_stall_mark}")
            else: print("  Waiting for robot data...")
            print(f"💾 Writer Queue: {writer.q.qsize()}")
            last_status_print = now_wall

        # --- 평균 FPS 출력 ---
        if now_wall - last_fps_print >= FPS_PERIOD:
            elapsed = now_wall - last_fps_print
            line = " | ".join([f"{k}:{cam_cnt[k]/elapsed:.1f}fps" for k in sorted(cam_cnt)])
            print("📊 Camera Avg:", line)
            cam_cnt = defaultdict(int)
            last_fps_print = now_wall

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

finally:
    print("\n--- Final Summary & Cleanup ---")
    # --- [수정] 최종 로봇 데이터 저장 (항상 'w' 모드 사용) ---
    # robot_data_buffer가 비어있지 않은 경우에만 저장 함수 호출
    if robot_data_buffer:
        save_robot_data_to_csv(robot_data_buffer, ROBOT_CSV_PATH)
    else:
        print("💾 [Robot Save] No robot data accumulated to save.")
    # --- [수정 끝] ---

    # 카메라 요약
    print("\n📷 Final Camera Summary:")
    for k in sorted(cam_latest_recv.keys()): data = cam_latest_recv[k]; print(f"  {k:<25} | cap_last:{data['capture_time']:.3f} | send_last:{data['send_time']:.3f} | recv_last:{data['recv_time']:.3f} | fail:{cam_fail_count.get(k,0)}")

    # 로봇 최종 상태
    print("\n🤖 Final Robot State:")
    if robot_latest_state: r_data = robot_latest_state; print(f"  ori:{r_data['origin_ts']:.3f} | send:{r_data['send_ts']:.3f} | recv:{r_data['recv_time']:.3f} | J1:{r_data['joints'][0]:.1f} | Px:{r_data['pose'][0]:.1f}")
    else: print("  No robot data received.")

    # 종료 처리
    writer.stop()
    writer.join()
    # ZMQ 소켓 정리
    print("🧹 Cleaning up ZMQ sockets...")
    poller.unregister(cam_sock)
    poller.unregister(robot_sock)
    cam_sock.close()
    robot_sock.close()
    if not ctx.closed:
        ctx.term()
    print("✅ Receiver shutdown complete.")