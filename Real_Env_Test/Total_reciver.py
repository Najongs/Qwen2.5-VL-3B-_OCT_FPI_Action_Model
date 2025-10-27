import os, time, json, cv2, zmq, numpy as np
import threading
from queue import Queue, Empty
from collections import defaultdict, deque # deque 추가
from datetime import datetime
import struct
import csv
import signal # signal 추가
import socket
import gc

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
# 파일 경로 정의
ROBOT_CSV_PATH = os.path.join(BASE_DIR, f"robot_state_{session_time}.csv")
SENSOR_NPZ_BASE_FILENAME = f"sensor_data_{session_time}" # NPZ 파일 기본 이름 (타임스탬프는 저장 시 추가)

# ==============================
# 2️⃣ 설정 및 파라미터
# ==============================
# 카메라/로봇 공통
ZED_SERIAL_TO_VIEW = {"41182735": "View1", "49429257": "View2", "44377151": "View3", "49045152": "View4"}
OAK_KEYWORD = "OAK"
STALL_SEC = 5.0
WARN_NET_MS = 200.0
WARN_TOTAL_MS = 500.0  # 👈 이 줄을 추가합니다. (500ms는 예시 값이며, 환경에 맞게 조정 가능)
STATUS_PERIOD = 1.0
FPS_PERIOD = 5.0

# 로봇 ZMQ 설정
ZMQ_ROBOT_PUB_ADDRESS = "10.130.41.111" # 송신측 IP
ZMQ_ROBOT_PUB_PORT = 5556
ZMQ_ROBOT_TOPIC = b"robot_state"
ROBOT_PAYLOAD_FORMAT = '<ddf12f' # ts, send_ts, force, 6x joints, 6x pose
ROBOT_PAYLOAD_SIZE = struct.calcsize(ROBOT_PAYLOAD_FORMAT) # 68 bytes

# 카메라 ZMQ 설정
ZMQ_CAM_PULL_PORT = 5555

# UDP 센서 설정
SENSOR_UDP_PORT = 9999
SENSOR_UDP_IP = "0.0.0.0" # 모든 인터페이스에서 수신
SENSOR_BUFFER_SIZE = 4 * 1024 * 1024
SENSOR_NXZRt = 1025
# 센서 패킷 구조 (C++ 송신측과 일치, 4120B)
SENSOR_PACKET_HEADER_FORMAT = '<ddf' # ts, send_ts, force
SENSOR_PACKET_HEADER_SIZE = struct.calcsize(SENSOR_PACKET_HEADER_FORMAT)  # 20B
SENSOR_ALINE_FORMAT = f'<{SENSOR_NXZRt}f'
SENSOR_ALINE_SIZE = struct.calcsize(SENSOR_ALINE_FORMAT)                  # 4100B
SENSOR_TOTAL_PACKET_SIZE = SENSOR_PACKET_HEADER_SIZE + SENSOR_ALINE_SIZE         # 4120B
SENSOR_CALIBRATION_COUNT = 50 # 시계 오차 보정 샘플 수

# 저장 제어 플래그
START_SAVE_FLAG = threading.Event() # 실제 저장을 시작할지 결정하는 전역 플래그
ROBOT_RECEIVED_FIRST = False        # 로봇 첫 수신 확인
SENSOR_RECEIVED_FIRST = False       # 센서 첫 수신 확인
CAM_RECEIVED_ALL_VIEWS = False      # 모든 카메라 뷰(View1~View5) 첫 수신 확인

REQUIRED_VIEWS = set(ZED_SERIAL_TO_VIEW.values()) | {"View5"} # {'View1', 'View2', 'View3', 'View4', 'View5'}
CAM_RECEIVED_VIEWS = set()

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
            try: self.q.put_nowait((path, img))
            except: print(f"[Writer] Queue full, dropping frame: {path}")

    def run(self):
        while True:
            try:
                path, img = self.q.get(timeout=0.1)
                try: cv2.imwrite(path, img)
                except Exception as e: print(f"[Writer] Error saving {path}: {e}")
                finally: self.q.task_done()
            except Empty:
                if self.stop_flag.is_set() and self.q.empty(): break
                continue

    def stop(self):
        print(f"🕒 Flushing remaining {self.q.qsize()} images before shutdown...")
        self.stop_flag.set()
        self.q.join()
        print("🛑 Writer thread stopped.")

# ==============================
# 4️⃣ ZMQ / UDP 소켓 및 Poller 설정
# ==============================
ctx = zmq.Context.instance()
cam_sock = ctx.socket(zmq.PULL)
cam_sock.setsockopt(zmq.RCVHWM, 5000); cam_sock.setsockopt(zmq.RCVBUF, 8 * 1024 * 1024) # 버퍼 증가
cam_bind_addr = f"tcp://0.0.0.0:{ZMQ_CAM_PULL_PORT}"; cam_sock.bind(cam_bind_addr)
print(f"✅ Camera PULL listening on {cam_bind_addr}")

robot_sock = ctx.socket(zmq.SUB)
robot_sock.setsockopt(zmq.RCVHWM, 100)
robot_connect_addr = f"tcp://{ZMQ_ROBOT_PUB_ADDRESS}:{ZMQ_ROBOT_PUB_PORT}"; robot_sock.connect(robot_connect_addr)
robot_sock.subscribe(ZMQ_ROBOT_TOPIC)
print(f"✅ Robot SUB connected to {robot_connect_addr} (Topic: '{ZMQ_ROBOT_TOPIC.decode()}')")

poller = zmq.Poller()
poller.register(cam_sock, zmq.POLLIN)
poller.register(robot_sock, zmq.POLLIN)

writer = AsyncImageWriter()
writer.start()

# ==============================
# 5️⃣ 공유 상태 변수
# ==============================
# 카메라 상태
cam_cnt = defaultdict(int); cam_fail_count = defaultdict(int); cam_last_ts = {}
cam_delta_dict = {}; cam_latest_recv = {}; cam_last_recv_wall = {}; cam_save_dir_cache = {}
# 로봇 상태
robot_cnt = 0; robot_latest_state = {}; robot_last_recv_wall = 0.0
robot_data_buffer = [] # CSV 저장을 위한 메모리 버퍼
# 센서 상태 (UDP 스레드와 공유)
sensor_lock = threading.Lock() # deque 접근용 락 (여기선 미사용)
sensor_save_lock = threading.Lock() # save_buffer 접근용 락
sensor_save_buffer = [] # NPZ 저장을 위한 메모리 버퍼
sensor_latest_status = { # 상태 로깅용
    "avg_latency": 0.0, "batch_count": 0, "packet_count": 0,
    "buffer_size": 0, "last_recv_wall": 0.0
}
sensor_clock_offset_s = None
sensor_calibration_samples = []

# 공통 상태
stop_event = threading.Event() # 모든 스레드 종료용
t0 = time.time()
last_status_print = time.time()
last_fps_print = time.time()

# ==============================
# 6️⃣ 유틸리티 함수
# ==============================
def get_save_dir_for_cam(cam_name: str):
    if cam_name in cam_save_dir_cache: return cam_save_dir_cache[cam_name]
    cam_lower = cam_name.lower(); view_dir = os.path.join(BASE_DIR, "Unknown")
    if "zed" in cam_lower:
        for serial, view_name in ZED_SERIAL_TO_VIEW.items():
            if serial in cam_lower: view_dir = os.path.join(BASE_DIR, view_name); break
    elif OAK_KEYWORD.lower() in cam_lower: view_dir = os.path.join(BASE_DIR, "View5")
    if "left" in cam_lower: save_dir = os.path.join(view_dir, "left")
    elif "right" in cam_lower: save_dir = os.path.join(view_dir, "right")
    else: save_dir = view_dir
    os.makedirs(save_dir, exist_ok=True); cam_save_dir_cache[cam_name] = save_dir
    return save_dir

def save_robot_data_to_csv(data_list, filepath):
    if not data_list: print("💾 [Robot Save] No robot data to save."); return
    mode = 'w'; print(f"💾 Saving {len(data_list)} robot states to {filepath} (Mode: {mode})")
    try:
        with open(filepath, mode, newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["recv_timestamp", "origin_timestamp", "send_timestamp", "force_placeholder", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "pose_x", "pose_y", "pose_z", "pose_a", "pose_b", "pose_r"])
            w.writerows(data_list)
        print(f"💾✅ Saved robot data successfully.")
    except Exception as e: print(f"[ERROR] Failed to save robot data to CSV: {e}")

def save_sensor_data_to_npz(data_list, base_filename):
    """ NPZ 파일로 저장. 파일명에 첫 타임스탬프 포함 """
    if not data_list: print("💾 [Sensor Save] No sensor data to save."); return
    try:
        first_ts_int = int(data_list[0]['timestamp'])
        filename = f"{base_filename}_{first_ts_int}.npz"
        filepath = os.path.join(BASE_DIR, filename)
        print(f"\n💾 Saving {len(data_list)} sensor records to {filepath}...")
        timestamps = np.array([d['timestamp'] for d in data_list], dtype=np.float64)
        send_timestamps = np.array([d['send_timestamp'] for d in data_list], dtype=np.float64)
        forces = np.array([d['force'] for d in data_list], dtype=np.float32)
        alines = np.array([d['aline'] for d in data_list], dtype=np.float32)
        np.savez(filepath, timestamps=timestamps, send_timestamps=send_timestamps, forces=forces, alines=alines) # 압축 없이 저장
        print(f"💾✅ Saved sensor data successfully! ({filepath})")
        print(f"  - timestamps: {timestamps.shape}, forces: {forces.shape}, alines: {alines.shape}")
    except Exception as e: print(f"[ERROR] Failed to save sensor data to NPZ ({filepath}): {e}")

def handle_sigint(sig, frame):
    print("\n🛑 Ctrl+C detected — Signaling stop to all threads...")
    stop_event.set()

def get_view_name_from_cam(cam_name: str) -> str:
    """ 카메라 이름에서 View1 ~ View5 이름을 추출합니다. """
    cam_lower = cam_name.lower()
    for serial, view_name in ZED_SERIAL_TO_VIEW.items():
        if serial in cam_lower: return view_name
    if OAK_KEYWORD.lower() in cam_lower: return "View5"
    return "Unknown" # 알 수 없는 카메라는 무시

def check_all_ready():
    """ 로봇, 센서, 카메라가 모두 첫 데이터를 수신했는지 확인합니다. """
    if ROBOT_RECEIVED_FIRST and SENSOR_RECEIVED_FIRST and CAM_RECEIVED_ALL_VIEWS:
        if not START_SAVE_FLAG.is_set():
            t_start = time.time()
            START_SAVE_FLAG.set()
            print("\n" + "#"*80)
            print(f"!!! 🚀 ALL SYSTEMS READY! STARTING DATA COLLECTION AND SAVE! (t={t_start:.3f})")
            print("#"*80 + "\n")

signal.signal(signal.SIGINT, handle_sigint) # 시그널 핸들러 설정

# ==============================
# 7️⃣ UDP 센서 수신 스레드
# ==============================
# ▼▼▼ [수정된 섹션] ▼▼▼
# (데이터 누락(손실) 버그가 수정된 버전입니다)
def sensor_udp_receiver_thread():
    # SENSOR_RECEIVED_FIRST를 함수 내에서 수정하므로 global 선언
    global sensor_clock_offset_s, sensor_latest_status, sensor_save_buffer, SENSOR_RECEIVED_FIRST

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 소켓 옵션 설정 (버퍼 크기 증가 - 설정 파일의 SENSOR_BUFFER_SIZE 사용)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SENSOR_BUFFER_SIZE) # 4MB 권장
        sock.bind((SENSOR_UDP_IP, SENSOR_UDP_PORT))
        sock.settimeout(1.0)
        print(f"✅ Sensor UDP Receiver started on port {SENSOR_UDP_PORT} (Single Packet Mode)")
    except Exception as e:
        print(f"[ERROR] Failed to bind UDP socket on port {SENSOR_UDP_PORT}: {e}")
        stop_event.set(); return

    print(f"⏳ Calibrating sensor clock offset using first {SENSOR_CALIBRATION_COUNT} batches...")
    
    # [수정] 2단계(헤더/페이로드) 버퍼링 로직 모두 제거
    last_log_time = time.time()
    batch_count_sec = 0; packet_count_sec = 0; latency_samples_sec = []

    while not stop_event.is_set():
        try:
            # [수정] 하나의 데이터그램을 통째로 수신
            # SENSOR_BUFFER_SIZE는 C++의 최대 전송 크기(약 62KB)보다 커야 함
            data, addr = sock.recvfrom(SENSOR_BUFFER_SIZE)
        except socket.timeout: continue
        except Exception as e:
            if stop_event.is_set(): break
            print(f"[UDP Sensor] Receive error: {e}"); continue
        
        recv_time = time.time()

        # [수정] 수신된 데이터그램은 최소 (헤더 4B + 최소 1패킷 4120B) 보다는 커야 함
        if len(data) < SENSOR_TOTAL_PACKET_SIZE:
            print(f"[WARN] Sensor UDP: Runt packet received ({len(data)}B). Discarding.")
            continue
            
        try:
            # 1. 데이터그램의 *맨 앞* 4바이트에서 헤더(패킷 수)를 엽니다.
            num_packets_in_batch = struct.unpack('<I', data[:4])[0]
            
            # 2. 크기가 일치하는지 확인합니다.
            #    (예상 크기 = 4B 헤더 + (패킷 수 * 패킷당 크기))
            expected_total_size = 4 + (num_packets_in_batch * SENSOR_TOTAL_PACKET_SIZE)
            actual_total_size = len(data)
            
            if actual_total_size != expected_total_size:
                print(f"[WARN] Sensor UDP: Corrupt packet. Header says {num_packets_in_batch} pkts (expected {expected_total_size}B) but got {actual_total_size}B.")
                continue
                
            if num_packets_in_batch == 0:
                continue # 0개짜리 배치는 무시

            # 3. 페이로드 처리 (데이터의 4바이트 *이후*부터)
            records = []
            mv = memoryview(data)[4:] # [수정] 4바이트 오프셋 적용
            offset = 0
            last_ts_in_batch, last_send_ts_in_batch = 0.0, 0.0
            
            for _ in range(num_packets_in_batch):
                # (패킷 파싱 로직은 이전과 동일)
                header = mv[offset:offset + SENSOR_PACKET_HEADER_SIZE]
                ts, send_ts, force = struct.unpack(SENSOR_PACKET_HEADER_FORMAT, header); offset += SENSOR_PACKET_HEADER_SIZE
                aline_bytes = mv[offset:offset + SENSOR_ALINE_SIZE]
                aline = np.frombuffer(aline_bytes, dtype=np.float32).copy(); offset += SENSOR_ALINE_SIZE
                records.append({"timestamp": ts, "send_timestamp": send_ts, "force": float(force), "aline": aline})
                last_ts_in_batch, last_send_ts_in_batch = ts, send_ts
                
        except Exception as e:
            # 파싱 실패 시(데이터 손상 등)
            print(f"[ERROR] Sensor UDP unpack failed (data len {len(data)}): {e}")
            continue # 이 손상된 패킷은 버립니다.

        # ⚠️ START_SAVE_FLAG가 설정되었을 때만 버퍼에 추가합니다.
        if START_SAVE_FLAG.is_set():
            with sensor_save_lock: sensor_save_buffer.extend(records)
        
        # --- 클럭 오프셋 보정 및 지연 시간 계산 로직 ---
        # (last_send_ts_in_batch가 0인 경우(패킷 0개) 방지)
        if num_packets_in_batch > 0:
            net_plus_offset_s = recv_time - last_send_ts_in_batch
            if sensor_clock_offset_s is None:
                sensor_calibration_samples.append(net_plus_offset_s)
                if len(sensor_calibration_samples) >= SENSOR_CALIBRATION_COUNT:
                    sensor_clock_offset_s = np.mean(sensor_calibration_samples)
                    print("\n" + "="*80 + f"\n✅ Sensor Clock Offset Calibrated: {sensor_clock_offset_s * 1000:.1f} ms\n" + "="*80 + "\n")
                    
                    # 1. 🚨 센서 최초 수신 (보정 완료) 확인 🚨
                    if not SENSOR_RECEIVED_FIRST:
                        SENSOR_RECEIVED_FIRST = True
                        print("🔬 Sensor: Calibration complete. Checking readiness...")
                        check_all_ready()
                        
                else: print(f"⏳ Sensor Calibrating... ({len(sensor_calibration_samples)}/{SENSOR_CALIBRATION_COUNT})", end='\r')
            else:
                queue_delay_cpp_ms = (last_send_ts_in_batch - last_ts_in_batch) * 1000
                net_delay_ms = (net_plus_offset_s - sensor_clock_offset_s) * 1000
                corrected_total_delay_ms = queue_delay_cpp_ms + net_delay_ms
                batch_count_sec += 1; packet_count_sec += num_packets_in_batch; latency_samples_sec.append(corrected_total_delay_ms)
    
    # ▲▲▲ [수정된 로직 끝] ▲▲▲

    # --- 1초마다 로깅 및 상태 업데이트 ---
    current_time = time.time()
    if current_time - last_log_time >= 1.0: # STATUS_PERIOD 대신 1.0 사용
        current_status = {}
        if sensor_clock_offset_s is not None:
            avg_lat = np.mean(latency_samples_sec) if latency_samples_sec else 0.0
            with sensor_save_lock: current_buffer_size = len(sensor_save_buffer)
            current_status = {
                "avg_latency": avg_lat, "batch_count": batch_count_sec, "packet_count": packet_count_sec,
                "buffer_size": current_buffer_size,
                "last_recv_wall": current_time if batch_count_sec > 0 else sensor_latest_status.get("last_recv_wall", 0.0) # 데이터 수신 시에만 갱신
            }
            latency_samples_sec.clear(); batch_count_sec = 0; packet_count_sec = 0
        else: # 보정 중일 때
                current_status = sensor_latest_status.copy() # 이전 상태 복사
                # Stall 감지용으로 last_recv_wall을 현재 시간으로 갱신
                # (실제 데이터 수신 여부와 관계없이 스레드가 돌고 있음을 의미)
                current_status["last_recv_wall"] = current_time

        # 전역 변수 업데이트 (메인 스레드에서 읽기 위함)
        sensor_latest_status = current_status
        last_log_time = current_time

    sock.close()
    print("🛑 Sensor UDP Receiver thread stopped.")
# ▲▲▲ [수정된 섹션] ▲▲▲

# ==============================
# 8️⃣ 메인 수신 루프 (통합됨)
# ==============================
try:
    sensor_thread = threading.Thread(target=sensor_udp_receiver_thread, daemon=True)
    sensor_thread.start()

    while not stop_event.is_set():
        try:
            socks = dict(poller.poll(timeout=100))
        except KeyboardInterrupt: stop_event.set(); print("\n🛑 Ctrl+C detected in main loop."); break
        except Exception as e: print(f"[WARN] Main poller.poll error: {e}"); time.sleep(0.1); continue

        now_wall = time.time()

        # --- 카메라 ZMQ 소켓 처리 ---
        if cam_sock in socks and socks[cam_sock] == zmq.POLLIN:
            while True:
                try:
                    parts = cam_sock.recv_multipart(zmq.DONTWAIT)
                    if len(parts) < 2: continue
                    meta_raw, jpg = parts[0], parts[1]
                    if not jpg or len(jpg) < 1000: continue
                    try: meta = json.loads(meta_raw.decode("utf-8"))
                    except: continue

                    cam = meta.get("camera", "unknown"); ts = float(meta.get("timestamp", 0.0)); send_time = float(meta.get("send_time", 0.0))
                    net_delay = (now_wall - send_time) if send_time > 0 else 0.0; total_delay = (now_wall - ts) if ts > 0 else 0.0
                    
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if img is None: cam_fail_count[cam] += 1; continue
                    
                    if not CAM_RECEIVED_ALL_VIEWS: # 모든 뷰가 수신되기 전까지 확인
                        view_name = get_view_name_from_cam(cam)
                        if view_name and view_name != "Unknown":
                            CAM_RECEIVED_VIEWS.add(view_name) # 수신된 뷰 기록
                            
                            if len(CAM_RECEIVED_VIEWS) == len(REQUIRED_VIEWS): # 모든 필수 뷰가 들어왔는지 확인
                                CAM_RECEIVED_ALL_VIEWS = True
                                print(f"✨ Cameras: All {len(REQUIRED_VIEWS)} views received ({', '.join(sorted(CAM_RECEIVED_VIEWS))}). Checking readiness...")
                                check_all_ready() # 로봇/센서 상태와 최종 확인
                    # ----------------------------------------------------
                    
                    # 상태 업데이트 (저장 여부와 관계없이 실시간으로 갱신)
                    cam_cnt[cam] += 1
                    cam_latest_recv[cam] = {"capture_time": ts, "send_time": send_time, "recv_time": now_wall, "net_delay": net_delay, "total_delay": total_delay}
                    cam_last_recv_wall[cam] = now_wall
                    
                    # ⚠️ NameError 방지를 위해 delta를 안전하게 초기화
                    delta = -1.0 
                    if cam in cam_last_ts and ts > 0: 
                        delta = ts - cam_last_ts[cam]
                        if delta > 0: cam_delta_dict[cam] = delta
                    if ts > 0: cam_last_ts[cam] = ts
                    
                    # 2. 💾 START_SAVE_FLAG가 설정되었을 때만 이미지 저장 요청
                    if START_SAVE_FLAG.is_set():
                        save_dir = get_save_dir_for_cam(cam)
                        filename = f"{cam}_{ts:.3f}.jpg" if ts > 0 else f"{cam}_{now_wall:.3f}.jpg"
                        writer.submit(os.path.join(save_dir, filename), img)

                except zmq.Again: break
                except Exception as e: print(f"[ERROR] Cam processing error: {e}"); break

        # --- 로봇 ZMQ 소켓 처리 ---
        if robot_sock in socks and socks[robot_sock] == zmq.POLLIN:
            while True:
                try:
                    parts = robot_sock.recv_multipart(zmq.DONTWAIT)
                    if len(parts) != 2: continue
                    topic, payload = parts[0], parts[1]
                    if topic != ZMQ_ROBOT_TOPIC or len(payload) != ROBOT_PAYLOAD_SIZE: continue
                    try:
                        unpacked_data = struct.unpack(ROBOT_PAYLOAD_FORMAT, payload)
                        
                        origin_ts, send_ts, force_pl = unpacked_data[0:3]
                        joints, pose = unpacked_data[3:9], unpacked_data[9:15]
                        
                        # 1. 🚨 로봇 최초 수신 확인 🚨
                        if not ROBOT_RECEIVED_FIRST:
                            ROBOT_RECEIVED_FIRST = True
                            print("🤖 Robot: First data received. Checking readiness...")
                            check_all_ready() 
                        
                        robot_cnt += 1
                        robot_latest_state = {"recv_time": now_wall, "origin_ts": origin_ts, "send_ts": send_ts, "joints": joints, "pose": pose, "net_delay": now_wall - send_ts, "total_delay": now_wall - origin_ts}
                        robot_last_recv_wall = now_wall
                        
                        # 2. 💾 START_SAVE_FLAG가 설정되었을 때만 데이터 저장
                        if START_SAVE_FLAG.is_set():
                            robot_data_buffer.append([now_wall] + list(unpacked_data))
                            
                    except struct.error: continue # 언패킹 에러 시 건너뜀
                except zmq.Again: break
                except Exception as e: print(f"[ERROR] Robot processing error: {e}"); break
                
        # --- 실시간 상태 모니터링 ---
        if now_wall - last_status_print >= STATUS_PERIOD:
            print(f"\n--- Status Update ({datetime.now().strftime('%H:%M:%S')}) ---")
            print("📷 Cameras:")
            if not cam_latest_recv: print("  Waiting for camera data...")
            for k in sorted(cam_latest_recv.keys()): data = cam_latest_recv[k]; d = cam_delta_dict.get(k, 0.0); fps = (1.0 / d) if d > 0 else 0.0; net_ms = data['net_delay'] * 1000.0; total_ms = data['total_delay'] * 1000.0; net_flag = "⚠️" if net_ms > WARN_NET_MS else " "; total_flag = "⚠️" if total_ms > WARN_TOTAL_MS else " "; stall_mark = " ⛔STALLED" if (now_wall - cam_last_recv_wall.get(k, 0.0)) >= STALL_SEC else ""; fail_mark = f" (fail:{cam_fail_count.get(k,0)})" if cam_fail_count.get(k,0) > 0 else ""; print(f"  {k:<25} | cap:{data['capture_time']:.3f} | send:{data['send_time']:.3f} | recv:{data['recv_time']:.3f} | Δnet:{net_ms:6.1f}ms{net_flag} | Δtotal:{total_ms:6.1f}ms{total_flag} | {fps:5.2f}fps{stall_mark}{fail_mark}")

            print("🤖 Robot:")
            if robot_latest_state: r_data = robot_latest_state; r_net_ms = r_data['net_delay'] * 1000.0; r_total_ms = r_data['total_delay'] * 1000.0; r_net_flag = "⚠️" if r_net_ms > WARN_NET_MS else " "; r_total_flag = "❓" if abs(r_total_ms) > WARN_TOTAL_MS else " "; r_stall_mark = " ⛔STALLED" if (now_wall - robot_last_recv_wall) >= STALL_SEC else ""; print(f"  {'Robot State':<25} | ori:{r_data['origin_ts']:.3f} | send:{r_data['send_ts']:.3f} | recv:{r_data['recv_time']:.3f} | Δnet:{r_net_ms:6.1f}ms{r_net_flag} | Δtotal:{r_total_ms:6.1f}ms{r_total_flag} | J1:{r_data['joints'][0]:<6.1f} Px:{r_data['pose'][0]:<7.1f}{r_stall_mark}")
            else: print("  Waiting for robot data...")

            print("🔬 Sensor (UDP):")
            s_status = sensor_latest_status # 락 없이 최신 상태 읽기 (원자적)
            s_avg_lat = s_status.get('avg_latency', 0.0)
            s_buffer = s_status.get('buffer_size', 0)
            s_last_recv = s_status.get('last_recv_wall', 0.0)
            s_stall_mark = " ⛔STALLED" if (now_wall - s_last_recv) >= STALL_SEC else ""
            if sensor_clock_offset_s is not None:
                 print(f"  {'Sensor State':<25} | Last Recv: {datetime.fromtimestamp(s_last_recv).strftime('%H:%M:%S')} | Avg Lat: {s_avg_lat:6.1f}ms | SaveBuf: {s_buffer:<6d}{s_stall_mark}")
            else:
                 print(f"  Sensor Calibrating... Last Recv: {datetime.fromtimestamp(s_last_recv).strftime('%H:%M:%S') if s_last_recv > 0 else 'N/A'}{s_stall_mark}")

            print(f"💾 Writer Queue: {writer.q.qsize()}")
            last_status_print = now_wall

        # --- 평균 FPS 출력 ---
        if now_wall - last_fps_print >= FPS_PERIOD:
            elapsed = now_wall - last_fps_print
            if cam_cnt:
                line = " | ".join([f"{k}:{cam_cnt[k]/elapsed:.1f}fps" for k in sorted(cam_cnt)])
                print("📊 Camera Avg FPS:", line)
            cam_cnt = defaultdict(int) # Reset camera count
            # 센서 FPS/bps 등 추가 가능
            last_fps_print = now_wall

finally:
    print("\n--- Final Summary & Cleanup ---")
    if not stop_event.is_set():
        print("🛑 Signaling stop event...")
        stop_event.set()

    # 스레드 종료 대기 (UDP 먼저)
    print("⏳ Waiting for Sensor UDP thread to finish...")
    if 'sensor_thread' in locals() and sensor_thread.is_alive():
        sensor_thread.join(timeout=5.0)
        if sensor_thread.is_alive(): print("[WARN] Sensor UDP thread did not exit cleanly.")
        else: print("✅ Sensor UDP thread finished.")

    # 최종 데이터 저장
    save_robot_data_to_csv(robot_data_buffer, ROBOT_CSV_PATH)
    robot_data_buffer.clear()

    # 센서 데이터 저장 (sensor_save_buffer 사용)
    with sensor_save_lock: # 저장 버퍼 접근 시 락 사용
        save_sensor_data_to_npz(list(sensor_save_buffer), SENSOR_NPZ_BASE_FILENAME) # 복사본 전달
        sensor_save_buffer.clear()

    # 최종 상태 요약
    print("\n📷 Final Camera Summary:")
    for k in sorted(cam_latest_recv.keys()): data = cam_latest_recv[k]; print(f"  {k:<25} | cap_last:{data['capture_time']:.3f} | send_last:{data['send_time']:.3f} | recv_last:{data['recv_time']:.3f} | fail:{cam_fail_count.get(k,0)}")
    print("\n🤖 Final Robot State:")
    if robot_latest_state: r_data = robot_latest_state; print(f"  ori:{r_data['origin_ts']:.3f} | send:{r_data['send_ts']:.3f} | recv:{r_data['recv_time']:.3f} | J1:{r_data['joints'][0]:.1f} | Px:{r_data['pose'][0]:.1f}")
    else: print("  No robot data received.")
    print("\n🔬 Final Sensor Status:")
    s_status = sensor_latest_status
    # print(f"  Avg Latency: {s_status.get('avg_latency', 0.0):.1f}ms | Buffer Size: {s_status.get('buffer_size', 0)} | Last Recv: {datetime.fromtimestamp(s_status.get('last_recv_wall', 0.0)).strftime('%Y-%m-%d %H:%M:%S') if s_status.get('last_recv_wall', 0.0) > 0 else 'N/A'}")
    print(f"  Avg Latency: {s_status.get('avg_latency', 0.0):.1f}ms | Buffer Size: {s_status.get('buffer_size', 0)} | Last Recv: {s_status.get('last_recv_wall', 0.0) if s_status.get('last_recv_wall', 0.0) > 0 else 'N/A'}")

    # 이미지 저장 스레드 종료 대기
    writer.stop()
    writer.join()

    print("♻️ Initiating final Garbage Collection...")
    gc.collect()
    print("✅ Garbage Collection finished.")
    
    # ZMQ 소켓 정리
    print("🧹 Cleaning up ZMQ sockets...")
    try:
        if cam_sock in poller.sockets: poller.unregister(cam_sock)
        if robot_sock in poller.sockets: poller.unregister(robot_sock)
        if not cam_sock.closed: cam_sock.close()
        if not robot_sock.closed: robot_sock.close()
        if not ctx.closed: ctx.term()
    except Exception as e: print(f"[WARN] Error during ZMQ cleanup: {e}")

    print("✅✅ Receiver shutdown complete.")