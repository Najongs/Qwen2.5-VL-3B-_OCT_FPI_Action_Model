import os, time, json, cv2, zmq, numpy as np
import threading
from queue import Queue
from collections import defaultdict
from datetime import datetime
import struct # <-- 로봇 데이터 언패킹 위해 추가

# ==============================
# 1️⃣ 세션 폴더 생성 (동일)
# ==============================
session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = f"./recv_all_{session_time}"
os.makedirs(BASE_DIR, exist_ok=True)
# ... (폴더 생성 코드 동일) ...
print(f"📁 Save directory: {BASE_DIR}")
# 로봇 데이터 저장 파일
ROBOT_CSV_PATH = os.path.join(BASE_DIR, f"robot_state_{session_time}.csv")

# ==============================
# 2️⃣ 설정 및 파라미터 (로봇 관련 추가)
# ==============================
ZED_SERIAL_TO_VIEW = { # ... (동일) ...
    "41182735": "View1", "49429257": "View2", "44377151": "View3", "49045152": "View4",
}
OAK_KEYWORD = "OAK"

# 모니터링 임계값 (동일)
STALL_SEC = 3.0
WARN_NET_MS = 150.0
WARN_TOTAL_MS = 1000.0 # 카메라 총 지연 기준
STATUS_PERIOD = 1.0
FPS_PERIOD = 2.0

# --- [로봇 ZMQ 설정 추가] ---
ZMQ_ROBOT_PUB_ADDRESS = "localhost" # 또는 로봇 스크립트 실행 IP
ZMQ_ROBOT_PUB_PORT = 5556
ZMQ_ROBOT_TOPIC = b"robot_state"
ROBOT_PAYLOAD_FORMAT = '<ddf12f' # ts, send_ts, force, 6x joints, 6x pose
ROBOT_PAYLOAD_SIZE = struct.calcsize(ROBOT_PAYLOAD_FORMAT) # 68 bytes

# --- [카메라 ZMQ 설정] ---
ZMQ_CAM_PULL_PORT = 5555

# ==============================
# 3️⃣ 비동기 이미지 저장 (동일)
# ==============================
class AsyncImageWriter(threading.Thread):
    # ... (코드 변경 없음) ...
    def __init__(self, max_queue=5000):
        super().__init__(daemon=True)
        self.q = Queue(max_queue)
        self.stop_flag = threading.Event()

    def submit(self, path, img):
        if not self.stop_flag.is_set():
            try: self.q.put_nowait((path, img))
            except: print(f"[Writer] Queue full, dropping frame: {path}")

    def run(self):
        while True:
            try: path, img = self.q.get(timeout=0.1)
            except:
                if self.stop_flag.is_set() and self.q.empty(): break
                continue
            try: cv2.imwrite(path, img)
            except Exception as e: print(f"[Writer] Error saving {path}: {e}")

    def stop(self):
        print(f"🕒 Flushing remaining {self.q.qsize()} images before shutdown...")
        self.stop_flag.set()
        while not self.q.empty(): time.sleep(0.1)
        print("🛑 Writer thread stopped.")

# ==============================
# 4️⃣ ZMQ 수신 설정 (수정됨 - 소켓 2개 + Poller)
# ==============================
ctx = zmq.Context.instance()

# 카메라 PULL 소켓
cam_sock = ctx.socket(zmq.PULL)
cam_sock.setsockopt(zmq.RCVHWM, 5000)
cam_sock.setsockopt(zmq.RCVBUF, 4 * 1024 * 1024)
cam_bind_addr = f"tcp://0.0.0.0:{ZMQ_CAM_PULL_PORT}"
cam_sock.bind(cam_bind_addr)
print(f"✅ Camera PULL listening on {cam_bind_addr}")

# 로봇 SUB 소켓
robot_sock = ctx.socket(zmq.SUB)
robot_sock.setsockopt(zmq.RCVHWM, 100) # 로봇 상태는 최신이 중요하므로 HWM 낮게
robot_connect_addr = f"tcp://{ZMQ_ROBOT_PUB_ADDRESS}:{ZMQ_ROBOT_PUB_PORT}"
robot_sock.connect(robot_connect_addr)
robot_sock.subscribe(ZMQ_ROBOT_TOPIC) # 특정 토픽만 구독
print(f"✅ Robot SUB connected to {robot_connect_addr} (Topic: '{ZMQ_ROBOT_TOPIC.decode()}')")

# Poller 생성 및 등록
poller = zmq.Poller()
poller.register(cam_sock, zmq.POLLIN)
poller.register(robot_sock, zmq.POLLIN)

writer = AsyncImageWriter()
writer.start()

# ==============================
# 5️⃣ 상태 변수 (로봇 관련 추가)
# ==============================
# 카메라 상태
cam_cnt = defaultdict(int)
cam_fail_count = defaultdict(int)
cam_last_ts = {}
cam_delta_dict = {}
cam_latest_recv = {}
cam_last_recv_wall = {}
cam_save_dir_cache = {}

# 로봇 상태
robot_cnt = 0
robot_latest_state = {}
robot_last_recv_wall = 0.0
robot_data_buffer = [] # CSV 저장을 위한 임시 버퍼

t0 = time.time()
last_status_print = time.time()
last_fps_print = time.time()

# ==============================
# 6️⃣ 유틸: 카메라별 디렉토리 캐시 (동일)
# ==============================
def get_save_dir_for_cam(cam_name: str):
    # ... (코드 변경 없음) ...
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

# --- [로봇 데이터 저장 함수 추가] ---
def save_robot_data_to_csv(data_list, filepath):
    """로봇 상태 데이터를 CSV 파일에 추가합니다."""
    is_new_file = not os.path.exists(filepath)
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if is_new_file:
                 writer.writerow([
                    "recv_timestamp", # 로컬 수신 시간
                    "origin_timestamp", # 송신측 clock.now()
                    "send_timestamp",   # 송신측 time.time()
                    "force_placeholder",
                    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
                    "pose_x", "pose_y", "pose_z", "pose_a", "pose_b", "pose_r"
                 ])
            writer.writerows(data_list)
        # print(f"💾 Saved {len(data_list)} robot states to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save robot data to CSV: {e}")


# ==============================
# 7️⃣ 수신 루프 (수정됨 - Poller 사용)
# ==============================
try:
    while True:
        try:
            # Poller로 이벤트 대기 (100ms 타임아웃)
            socks = dict(poller.poll(timeout=100))
        except KeyboardInterrupt:
            break # Ctrl+C 감지 시 루프 종료
        except Exception as e:
            print(f"[WARN] poller.poll error: {e}")
            time.sleep(0.1)
            continue

        now_wall = time.time() # 현재 시간 (여러 번 사용됨)

        # --- 카메라 소켓 처리 ---
        if cam_sock in socks and socks[cam_sock] == zmq.POLLIN:
            try:
                # DONTWAIT 사용: poll()이 데이터 있음을 보장
                parts = cam_sock.recv_multipart(zmq.DONTWAIT)

                if len(parts) < 2: print("[WARN] Cam multipart length < 2, skip"); continue
                meta_raw, jpg = parts[0], parts[1]
                if not jpg or len(jpg) < 5000: print(f"[WARN] Incomplete JPEG (len={len(jpg)})"); continue

                try: meta = json.loads(meta_raw.decode("utf-8"))
                except Exception as e: print(f"[WARN] Cam meta decode error: {e}"); continue

        cam = meta.get("camera", "unknown")
        ts = float(meta.get("timestamp", 0.0))
        send_time = float(meta.get("send_time", 0.0))
        recv_time = time.time()

        # ✨✨✨ 핵심 수정 부분: 핸드셰이크 패킷 (ts=0.0) 건너뛰기 ✨✨✨
        if ts == 0.0:
             print(f"⚪️ Received non-data message (ts=0.0) from {cam}, skipping.")
             continue # 다음 메시지 수신으로 넘어감
        # ✨✨✨ 수정 완료 ✨✨✨

                net_delay = (now_wall - send_time) if send_time > 0 else 0.0
                total_delay = (now_wall - ts) if ts > 0 else 0.0

                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    cam_fail_count[cam] += 1
                    print(f"[WARN] Cam JPEG decode failed for {cam} (fail #{cam_fail_count[cam]})")
                    continue

                cam_cnt[cam] += 1
                cam_latest_recv[cam] = {
                    "capture_time": ts, "send_time": send_time, "recv_time": now_wall,
                    "net_delay": net_delay, "total_delay": total_delay
                }
                cam_last_recv_wall[cam] = now_wall

                if cam in cam_last_ts and ts > 0:
                    delta = ts - cam_last_ts[cam]
                    if delta > 0: cam_delta_dict[cam] = delta
                if ts > 0: cam_last_ts[cam] = ts

                save_dir = get_save_dir_for_cam(cam)
                filename = f"{cam}_{ts:.3f}.jpg" if ts > 0 else f"{cam}_{now_wall:.3f}.jpg"
                writer.submit(os.path.join(save_dir, filename), img)

            except zmq.Again: # 이론상 발생 안 함 (poll 후 DONTWAIT)
                pass
            except Exception as e:
                 print(f"[ERROR] Unhandled exception during camera processing: {e}")


        # --- 로봇 소켓 처리 ---
        if robot_sock in socks and socks[robot_sock] == zmq.POLLIN:
            try:
                # DONTWAIT 사용
                parts = robot_sock.recv_multipart(zmq.DONTWAIT)

                if len(parts) != 2: print(f"[WARN] Robot multipart length != 2 (got {len(parts)}), skip"); continue
                topic, payload = parts[0], parts[1]

                if topic != ZMQ_ROBOT_TOPIC: print(f"[WARN] Unexpected robot topic: {topic}"); continue
                if len(payload) != ROBOT_PAYLOAD_SIZE:
                    print(f"[WARN] Robot payload size mismatch! Expected {ROBOT_PAYLOAD_SIZE}, got {len(payload)}")
                    continue

                try:
                    # 데이터 언패킹
                    unpacked_data = struct.unpack(ROBOT_PAYLOAD_FORMAT, payload)
                    origin_ts = unpacked_data[0]
                    send_ts = unpacked_data[1]
                    force_pl = unpacked_data[2]
                    joints = unpacked_data[3:9]
                    pose = unpacked_data[9:15]

                    robot_cnt += 1
                    robot_latest_state = {
                        "recv_time": now_wall,
                        "origin_ts": origin_ts,
                        "send_ts": send_ts,
                        "joints": joints,
                        "pose": pose,
                        "net_delay": now_wall - send_ts,
                        "total_delay": now_wall - origin_ts,
                    }
                    robot_last_recv_wall = now_wall
                    # CSV 저장을 위해 버퍼에 추가 (리스트 형태로)
                    robot_data_buffer.append([now_wall] + list(unpacked_data))

                except struct.error as e:
                    print(f"[WARN] Robot payload unpack error: {e}")
                    continue

            except zmq.Again:
                pass
            except Exception as e:
                 print(f"[ERROR] Unhandled exception during robot processing: {e}")


        # ==========================
        # 실시간 상태 모니터링 (수정됨)
        # ==========================
        if now_wall - last_status_print >= STATUS_PERIOD:
            print(f"\n--- Status Update ({datetime.now().strftime('%H:%M:%S')}) ---")
            # 카메라 상태 출력
            print("📷 Cameras:")
            for k in sorted(cam_latest_recv.keys()):
                data = cam_latest_recv[k]
                d = cam_delta_dict.get(k, 0.0)
                fps = (1.0 / d) if d > 0 else 0.0
                net_ms = data['net_delay'] * 1000.0
                total_ms = data['total_delay'] * 1000.0
                net_flag = "⚠️" if net_ms > WARN_NET_MS else " "
                total_flag = "⚠️" if total_ms > WARN_TOTAL_MS else " "
                stall_mark = " ⛔STALLED" if (now_wall - cam_last_recv_wall.get(k, 0.0)) >= STALL_SEC else ""
                fail_mark = f" (fail:{cam_fail_count.get(k,0)})" if cam_fail_count.get(k,0) > 0 else ""
                print(
                    f"  {k:<25} | "
                    f"cap:{data['capture_time']:.3f} | send:{data['send_time']:.3f} | recv:{data['recv_time']:.3f} | "
                    f"Δnet:{net_ms:6.1f}ms{net_flag} | Δtotal:{total_ms:6.1f}ms{total_flag} | "
                    f"{fps:5.2f}fps{stall_mark}{fail_mark}"
                )
            # 로봇 상태 출력
            print("🤖 Robot:")
            if robot_latest_state:
                 r_data = robot_latest_state
                 r_net_ms = r_data['net_delay'] * 1000.0
                 r_total_ms = r_data['total_delay'] * 1000.0
                 r_net_flag = "⚠️" if r_net_ms > WARN_NET_MS else " "
                 # 로봇은 clock 오차가 있으므로 total delay 경고는 참고용
                 r_total_flag = "❓" if abs(r_total_ms) > WARN_TOTAL_MS else " "
                 r_stall_mark = " ⛔STALLED" if (now_wall - robot_last_recv_wall) >= STALL_SEC else ""
                 print(
                    f"  {'Robot State':<25} | "
                    f"ori:{r_data['origin_ts']:.3f} | send:{r_data['send_ts']:.3f} | recv:{r_data['recv_time']:.3f} | "
                    f"Δnet:{r_net_ms:6.1f}ms{r_net_flag} | Δtotal:{r_total_ms:6.1f}ms{r_total_flag} | "
                    f"J1:{r_data['joints'][0]:<6.1f} Px:{r_data['pose'][0]:<7.1f}{r_stall_mark}"
                 )
            else:
                 print("  Waiting for robot data...")

            # Writer Queue Status
            print(f"💾 Writer Queue: {writer.q.qsize()}")
            last_status_print = now_wall

        # ==========================
        # 평균 FPS 출력 (카메라만)
        # ==========================
        if now_wall - last_fps_print >= FPS_PERIOD:
            elapsed = now_wall - last_fps_print
            line = " | ".join([f"{k}:{cam_cnt[k]/elapsed:.1f}fps" for k in sorted(cam_cnt)])
            print("📊 Camera Avg:", line)
            cam_cnt = defaultdict(int) # Reset camera count only
            last_fps_print = now_wall

        # ==========================
        # 로봇 데이터 주기적 저장 (예: 1000개 모이면 저장)
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
    print("\n--- Final Summary & Cleanup ---")
    # 최종 로봇 데이터 저장
    if robot_data_buffer:
        print(f"💾 Saving remaining {len(robot_data_buffer)} robot states...")
        save_robot_data_to_csv(list(robot_data_buffer), ROBOT_CSV_PATH)

    # 카메라 요약 (동일)
    print("\n📷 Final Camera Summary:")
    for k in sorted(cam_latest_recv.keys()):
        data = cam_latest_recv[k]; print(f"  {k:<25} | cap_last:{data['capture_time']:.3f} | send_last:{data['send_time']:.3f} | recv_last:{data['recv_time']:.3f} | fail:{cam_fail_count.get(k,0)}")

    # 로봇 최종 상태
    print("\n🤖 Final Robot State:")
    if robot_latest_state: r_data = robot_latest_state; print(f"  ori:{r_data['origin_ts']:.3f} | send:{r_data['send_ts']:.3f} | recv:{r_data['recv_time']:.3f} | J1:{r_data['joints'][0]:.1f} | Px:{r_data['pose'][0]:.1f}")
    else: print("  No robot data received.")

    # 종료 처리
    writer.stop()
    writer.join()
    poller.unregister(cam_sock)
    poller.unregister(robot_sock)
    cam_sock.close()
    robot_sock.close()
    ctx.term()
    print("✅ Receiver shutdown complete.")