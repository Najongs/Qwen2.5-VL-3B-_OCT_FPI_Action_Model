#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Fast Receiver - 18fps 완전 최적화
- 초고속 디스크 I/O
- 배치 저장
- 대용량 버퍼링
"""

import os, time, json, cv2, zmq, numpy as np, threading
from queue import Queue, Empty
from collections import defaultdict, deque
from datetime import datetime
import struct
import csv
import signal
import socket
import gc
from concurrent.futures import ThreadPoolExecutor

# ===================== 세션 폴더 생성 =====================
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

ROBOT_CSV_PATH = os.path.join(BASE_DIR, f"robot_state_{session_time}.csv")
SENSOR_NPZ_BASE_FILENAME = f"sensor_data_{session_time}"

# ===================== 설정 =====================
ZED_SERIAL_TO_VIEW = {
    "41182735": "View1", "49429257": "View2",
    "44377151": "View3", "49045152": "View4"
}
OAK_KEYWORD = "OAK"
STALL_SEC = 5.0
WARN_NET_MS = 200.0
WARN_TOTAL_MS = 500.0
STATUS_PERIOD = 1.0
FPS_PERIOD = 5.0

ZMQ_ROBOT_PUB_ADDRESS = "10.130.41.111"
ZMQ_ROBOT_PUB_PORT = 5556
ZMQ_ROBOT_TOPIC = b"robot_state"
ROBOT_PAYLOAD_FORMAT = '<ddf12f'
ROBOT_PAYLOAD_SIZE = struct.calcsize(ROBOT_PAYLOAD_FORMAT)

ZMQ_CAM_PULL_PORT = 5555

SENSOR_UDP_PORT = 9999
SENSOR_UDP_IP = "0.0.0.0"
SENSOR_BUFFER_SIZE = 4 * 1024 * 1024
SENSOR_NXZRt = 1025
SENSOR_PACKET_HEADER_FORMAT = '<ddf'
SENSOR_PACKET_HEADER_SIZE = struct.calcsize(SENSOR_PACKET_HEADER_FORMAT)
SENSOR_ALINE_FORMAT = f'<{SENSOR_NXZRt}f'
SENSOR_ALINE_SIZE = struct.calcsize(SENSOR_ALINE_FORMAT)
SENSOR_TOTAL_PACKET_SIZE = SENSOR_PACKET_HEADER_SIZE + SENSOR_ALINE_SIZE
SENSOR_CALIBRATION_COUNT = 50

START_SAVE_FLAG = threading.Event()
ROBOT_RECEIVED_FIRST = threading.Event()
SENSOR_RECEIVED_FIRST = threading.Event()
CAM_RECEIVED_ALL_VIEWS = threading.Event()

REQUIRED_VIEWS = set(ZED_SERIAL_TO_VIEW.values()) | {"View5"}
CAM_RECEIVED_VIEWS = set()
CAM_RECEIVED_VIEWS_LOCK = threading.Lock()

stop_event = threading.Event()

# ===================== 유틸리티 =====================
def get_view_name_from_cam(cam_name: str) -> str:
    cam_lower = cam_name.lower()
    for serial, view_name in ZED_SERIAL_TO_VIEW.items():
        if serial in cam_lower:
            return view_name
    if OAK_KEYWORD.lower() in cam_lower:
        return "View5"
    return "Unknown"

def get_save_dir_for_cam(cam_name: str, cache: dict):
    if cam_name in cache:
        return cache[cam_name]
    
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
    cache[cam_name] = save_dir
    return save_dir

def check_all_ready():
    if (ROBOT_RECEIVED_FIRST.is_set() and
        SENSOR_RECEIVED_FIRST.is_set() and
        CAM_RECEIVED_ALL_VIEWS.is_set()):
        
        if not START_SAVE_FLAG.is_set():
            START_SAVE_FLAG.set()
            print("\n" + "#"*80)
            print(f"🚀 ALL SYSTEMS READY! STARTING DATA SAVE!")
            print("#"*80 + "\n")

def handle_sigint(sig, frame):
    print("\n🛑 Ctrl+C detected")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)

def save_robot_data_to_csv(data_list, filepath):
    if not data_list:
        print("💾 [Robot] No data to save")
        return
    print(f"💾 Saving {len(data_list)} robot states to {filepath}")
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["recv_timestamp", "origin_timestamp", "send_timestamp",
                       "force_placeholder", "joint_1", "joint_2", "joint_3",
                       "joint_4", "joint_5", "joint_6", "pose_x", "pose_y",
                       "pose_z", "pose_a", "pose_b", "pose_r"])
            w.writerows(data_list)
        print(f"✅ Robot CSV saved")
    except Exception as e:
        print(f"[ERROR] Robot CSV save failed: {e}")

def save_sensor_data_to_npz(data_list, base_filename):
    if not data_list:
        print("💾 [Sensor] No data to save")
        return
    try:
        first_ts_int = int(data_list[0]['timestamp'])
        filename = f"{base_filename}_{first_ts_int}.npz"
        filepath = os.path.join(BASE_DIR, filename)
        print(f"💾 Saving {len(data_list)} sensor records to {filepath}...")
        
        timestamps = np.array([d['timestamp'] for d in data_list], dtype=np.float64)
        send_timestamps = np.array([d['send_timestamp'] for d in data_list], dtype=np.float64)
        forces = np.array([d['force'] for d in data_list], dtype=np.float32)
        alines = np.array([d['aline'] for d in data_list], dtype=np.float32)
        
        np.savez(filepath, timestamps=timestamps, send_timestamps=send_timestamps,
                forces=forces, alines=alines)
        print(f"✅ Sensor NPZ saved")
    except Exception as e:
        print(f"[ERROR] Sensor NPZ save failed: {e}")

# ===================== Ultra-Fast 이미지 저장 =====================
class UltraFastImageWriter:
    """
    초고속 이미지 저장 시스템
    - ThreadPoolExecutor로 병렬 저장
    - 배치 처리로 I/O 최적화
    - 대용량 버퍼링
    """
    def __init__(self, num_workers=20, max_queue=15000):
        self.q = Queue(max_queue)
        self.stop_flag = threading.Event()
        
        # ThreadPoolExecutor (I/O 병렬화)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # 통계
        self.lock = threading.Lock()
        self.submitted = 0
        self.saved = 0
        self.dropped = 0
        self.last_log = time.time()
        
        # 저장 스레드
        self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self.save_thread.start()
        
        print(f"✅ UltraFastImageWriter: {num_workers} workers, queue={max_queue}")
    
    def submit(self, path, img):
        """메인 스레드에서 호출 (논블로킹)"""
        if not self.stop_flag.is_set():
            try:
                self.q.put_nowait((path, img))
                with self.lock:
                    self.submitted += 1
            except:
                with self.lock:
                    self.dropped += 1
    
    def _save_loop(self):
        """저장 루프 (별도 스레드)"""
        batch = []
        last_batch_save = time.time()
        BATCH_SIZE = 25  # 25장씩 배치 저장
        BATCH_TIMEOUT = 0.05  # 50ms 타임아웃
        
        while not self.stop_flag.is_set():
            # 큐에서 수집
            try:
                item = self.q.get(timeout=0.005)
                batch.append(item)
            except Empty:
                pass
            
            # 배치 저장 조건
            should_save = (
                len(batch) >= BATCH_SIZE or
                (len(batch) > 0 and time.time() - last_batch_save > BATCH_TIMEOUT)
            )
            
            if should_save:
                # 배치를 복사 (원본 배치는 초기화)
                current_batch = batch[:]
                batch.clear()
                
                # 프로세스 풀에 저장 작업 제출
                for path, img in current_batch:
                    self.executor.submit(self._save_image, path, img)
                
                last_batch_save = time.time()
            
            # 통계
            self._maybe_log()
        
        # 종료 시 남은 배치 저장
        if batch:
            print(f"📦 Saving final batch: {len(batch)} images")
            for path, img in batch:
                self.executor.submit(self._save_image, path, img)
        
        with self.lock:
            print(f"📊 Final: submitted={self.submitted}, saved={self.saved}, dropped={self.dropped}")
    
    def _save_image(self, path, img):
        """실제 저장 (워커 스레드에서 실행)"""
        try:
            # JPEG 품질 85 (속도와 품질 밸런스)
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self.lock:
                self.saved += 1
        except Exception as e:
            print(f"[Writer] Error saving {path}: {e}")
    
    def _maybe_log(self):
        now = time.time()
        if now - self.last_log >= 3.0:
            with self.lock:
                elapsed = now - self.last_log
                saved_rate = self.saved / elapsed if elapsed > 0 else 0
                print(f"[Writer] q={self.q.qsize()} | save_rate={saved_rate:.1f}fps | "
                      f"submitted={self.submitted} | saved={self.saved} | dropped={self.dropped}")
                self.submitted = 0
                self.saved = 0
                self.last_log = now
    
    def stop(self):
        print(f"🕒 Stopping writer (queue: {self.q.qsize()})...")
        self.stop_flag.set()
        self.save_thread.join(timeout=5.0)
        
        # 워커 풀 종료 (모든 작업 완료 대기)
        print("⏳ Waiting for all save tasks to complete...")
        self.executor.shutdown(wait=True)
        
        print("✅ UltraFastImageWriter stopped")

# ===================== 카메라 수신 스레드 =====================
class CameraReceiver(threading.Thread):
    """카메라 ZMQ 전용 수신 + JPEG 디코딩"""
    def __init__(self, cam_sock, writer):
        super().__init__(daemon=True)
        self.cam_sock = cam_sock
        self.writer = writer
        self.lock = threading.Lock()
        
        # 통계
        self.cam_cnt = defaultdict(int)
        self.cam_fail_count = defaultdict(int)
        self.cam_last_ts = {}
        self.cam_delta_dict = {}
        self.cam_latest_recv = {}
        self.cam_last_recv_wall = {}
        self.cam_save_dir_cache = {}
        
        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.cam_sock, zmq.POLLIN)
        
        print("📷 CameraReceiver initialized")
    
    def get_stats(self):
        with self.lock:
            return {
                'cam_cnt': dict(self.cam_cnt),
                'cam_fail_count': dict(self.cam_fail_count),
                'cam_latest_recv': dict(self.cam_latest_recv),
                'cam_last_recv_wall': dict(self.cam_last_recv_wall),
                'cam_delta_dict': dict(self.cam_delta_dict)
            }
    
    def reset_counters(self):
        with self.lock:
            self.cam_cnt = defaultdict(int)
    
    def run(self):
        print("📷 CameraReceiver started (18fps mode)")
        
        while not stop_event.is_set():
            try:
                socks = dict(self.poller.poll(timeout=10))  # 10ms (빠른 반응)
            except Exception as e:
                if stop_event.is_set():
                    break
                time.sleep(0.01)
                continue
            
            if self.cam_sock not in socks:
                continue
            
            now_wall = time.time()
            
            # 논블로킹으로 모든 대기 메시지 처리
            while not stop_event.is_set():
                try:
                    parts = self.cam_sock.recv_multipart(zmq.DONTWAIT)
                    if len(parts) < 2:
                        continue
                    
                    meta_raw, jpg = parts[0], parts[1]
                    if not jpg or len(jpg) < 1000:
                        continue
                    
                    try:
                        meta = json.loads(meta_raw.decode("utf-8"))
                    except:
                        continue
                    
                    cam = meta.get("camera", "unknown")
                    ts = float(meta.get("timestamp", 0.0))
                    send_time = float(meta.get("send_time", 0.0))
                    
                    # JPEG 디코딩
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    
                    if img is None:
                        with self.lock:
                            self.cam_fail_count[cam] += 1
                        continue
                    
                    # 최초 수신 확인
                    if not CAM_RECEIVED_ALL_VIEWS.is_set():
                        view_name = get_view_name_from_cam(cam)
                        if view_name and view_name != "Unknown":
                            with CAM_RECEIVED_VIEWS_LOCK:
                                CAM_RECEIVED_VIEWS.add(view_name)
                                
                                if len(CAM_RECEIVED_VIEWS) == len(REQUIRED_VIEWS):
                                    CAM_RECEIVED_ALL_VIEWS.set()
                                    print(f"✨ Cameras: All {len(REQUIRED_VIEWS)} views received")
                                    check_all_ready()
                    
                    # 통계 업데이트
                    net_delay = (now_wall - send_time) if send_time > 0 else 0.0
                    total_delay = (now_wall - ts) if ts > 0 else 0.0
                    
                    # 네거티브 지연 처리
                    # 원인: 네트워크 지터, 시스템 타이밍, 시계 드리프트 (±5ms는 정상)
                    # 실제 네트워크 지연은 항상 양수이므로 음수는 0으로 처리
                    if net_delay < -0.002:  # -2ms 이하만 경고
                        # 심각한 시계 오차 (드물게 발생)
                        net_delay = 0.0
                    elif net_delay < 0:
                        # 정상 범위의 음수 (지터/타이밍) → 0으로 클램핑
                        net_delay = 0.0
                    
                    with self.lock:
                        self.cam_cnt[cam] += 1
                        self.cam_latest_recv[cam] = {
                            "capture_time": ts,
                            "send_time": send_time,
                            "recv_time": now_wall,
                            "net_delay": net_delay,
                            "total_delay": total_delay
                        }
                        self.cam_last_recv_wall[cam] = now_wall
                        
                        delta = -1.0
                        if cam in self.cam_last_ts and ts > 0:
                            delta = ts - self.cam_last_ts[cam]
                            if delta > 0:
                                self.cam_delta_dict[cam] = delta
                        if ts > 0:
                            self.cam_last_ts[cam] = ts
                    
                    # 저장 (락 밖에서)
                    if START_SAVE_FLAG.is_set():
                        save_dir = get_save_dir_for_cam(cam, self.cam_save_dir_cache)
                        filename = f"{cam}_{ts:.3f}.jpg" if ts > 0 else f"{cam}_{now_wall:.3f}.jpg"
                        self.writer.submit(os.path.join(save_dir, filename), img)
                
                except zmq.Again:
                    break
                except Exception as e:
                    print(f"[ERROR] Camera: {e}")
                    break
        
        print("🛑 CameraReceiver stopped")

# ===================== 로봇/센서 (기존과 동일) =====================
class RobotReceiver(threading.Thread):
    def __init__(self, robot_sock):
        super().__init__(daemon=True)
        self.robot_sock = robot_sock
        self.lock = threading.Lock()
        
        self.robot_cnt = 0
        self.robot_latest_state = {}
        self.robot_last_recv_wall = 0.0
        self.robot_data_buffer = []
        
        self.poller = zmq.Poller()
        self.poller.register(self.robot_sock, zmq.POLLIN)
        
        print("🤖 RobotReceiver initialized")
    
    def get_stats(self):
        with self.lock:
            return {
                'robot_cnt': self.robot_cnt,
                'robot_latest_state': dict(self.robot_latest_state),
                'robot_last_recv_wall': self.robot_last_recv_wall,
                'buffer_size': len(self.robot_data_buffer)
            }
    
    def get_data_buffer(self):
        with self.lock:
            return list(self.robot_data_buffer)
    
    def reset_counter(self):
        with self.lock:
            self.robot_cnt = 0
    
    def run(self):
        print("🤖 RobotReceiver started")
        
        while not stop_event.is_set():
            try:
                socks = dict(self.poller.poll(timeout=50))
            except Exception as e:
                if stop_event.is_set():
                    break
                time.sleep(0.01)
                continue
            
            if self.robot_sock not in socks:
                continue
            
            now_wall = time.time()
            
            while not stop_event.is_set():
                try:
                    parts = self.robot_sock.recv_multipart(zmq.DONTWAIT)
                    if len(parts) != 2:
                        continue
                    
                    topic, payload = parts[0], parts[1]
                    if topic != ZMQ_ROBOT_TOPIC or len(payload) != ROBOT_PAYLOAD_SIZE:
                        continue
                    
                    unpacked_data = struct.unpack(ROBOT_PAYLOAD_FORMAT, payload)
                    origin_ts, send_ts, force_pl = unpacked_data[0:3]
                    joints, pose = unpacked_data[3:9], unpacked_data[9:15]
                    
                    if not ROBOT_RECEIVED_FIRST.is_set():
                        ROBOT_RECEIVED_FIRST.set()
                        print("🤖 Robot: First data received")
                        check_all_ready()
                    
                    with self.lock:
                        self.robot_cnt += 1
                        self.robot_latest_state = {
                            "recv_time": now_wall,
                            "origin_ts": origin_ts,
                            "send_ts": send_ts,
                            "joints": joints,
                            "pose": pose,
                            "net_delay": now_wall - send_ts,
                            "total_delay": now_wall - origin_ts
                        }
                        self.robot_last_recv_wall = now_wall
                        
                        if START_SAVE_FLAG.is_set():
                            self.robot_data_buffer.append([now_wall] + list(unpacked_data))
                
                except zmq.Again:
                    break
                except Exception as e:
                    print(f"[ERROR] Robot: {e}")
                    break
        
        print("🛑 RobotReceiver stopped")

class SensorReceiver(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        
        self.sensor_save_buffer = []
        self.sensor_latest_status = {
            "avg_latency": 0.0,
            "batch_count": 0,
            "packet_count": 0,
            "buffer_size": 0,
            "last_recv_wall": 0.0
        }
        self.sensor_clock_offset_s = None
        self.sensor_calibration_samples = []
        
        print("🔬 SensorReceiver initialized")
    
    def get_stats(self):
        with self.lock:
            return dict(self.sensor_latest_status)
    
    def get_data_buffer(self):
        with self.lock:
            return list(self.sensor_save_buffer)
    
    def run(self):
        print("🔬 SensorReceiver started")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SENSOR_BUFFER_SIZE)
            sock.bind((SENSOR_UDP_IP, SENSOR_UDP_PORT))
            sock.settimeout(0.5)
            print(f"✅ Sensor UDP bound to port {SENSOR_UDP_PORT}")
        except Exception as e:
            print(f"[ERROR] Sensor UDP bind failed: {e}")
            stop_event.set()
            return
        
        print(f"⏳ Calibrating sensor clock (first {SENSOR_CALIBRATION_COUNT} batches)...")
        
        last_log_time = time.time()
        batch_count_sec = 0
        packet_count_sec = 0
        latency_samples_sec = []
        
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(SENSOR_BUFFER_SIZE)
            except socket.timeout:
                continue
            except Exception as e:
                if stop_event.is_set():
                    break
                print(f"[ERROR] Sensor UDP recv: {e}")
                continue
            
            recv_time = time.time()
            
            if len(data) < SENSOR_TOTAL_PACKET_SIZE:
                continue
            
            try:
                num_packets_in_batch = struct.unpack('<I', data[:4])[0]
                expected_total_size = 4 + (num_packets_in_batch * SENSOR_TOTAL_PACKET_SIZE)
                
                if len(data) != expected_total_size:
                    continue
                
                if num_packets_in_batch == 0:
                    continue
                
                records = []
                mv = memoryview(data)[4:]
                offset = 0
                last_ts_in_batch, last_send_ts_in_batch = 0.0, 0.0
                
                for _ in range(num_packets_in_batch):
                    header = mv[offset:offset + SENSOR_PACKET_HEADER_SIZE]
                    ts, send_ts, force = struct.unpack(SENSOR_PACKET_HEADER_FORMAT, header)
                    offset += SENSOR_PACKET_HEADER_SIZE
                    
                    aline_bytes = mv[offset:offset + SENSOR_ALINE_SIZE]
                    aline = np.frombuffer(aline_bytes, dtype=np.float32).copy()
                    offset += SENSOR_ALINE_SIZE
                    
                    records.append({
                        "timestamp": ts,
                        "send_timestamp": send_ts,
                        "force": float(force),
                        "aline": aline
                    })
                    last_ts_in_batch, last_send_ts_in_batch = ts, send_ts
            
            except Exception as e:
                print(f"[ERROR] Sensor unpack: {e}")
                continue
            
            if START_SAVE_FLAG.is_set():
                with self.lock:
                    self.sensor_save_buffer.extend(records)
            
            if num_packets_in_batch > 0:
                net_plus_offset_s = recv_time - last_send_ts_in_batch
                
                if self.sensor_clock_offset_s is None:
                    self.sensor_calibration_samples.append(net_plus_offset_s)
                    
                    if len(self.sensor_calibration_samples) >= SENSOR_CALIBRATION_COUNT:
                        self.sensor_clock_offset_s = np.mean(self.sensor_calibration_samples)
                        print(f"\n{'='*80}")
                        print(f"✅ Sensor Clock Offset: {self.sensor_clock_offset_s * 1000:.1f} ms")
                        print(f"{'='*80}\n")
                        
                        if not SENSOR_RECEIVED_FIRST.is_set():
                            SENSOR_RECEIVED_FIRST.set()
                            print("🔬 Sensor: Calibration complete")
                            check_all_ready()
                    else:
                        print(f"⏳ Calibrating... ({len(self.sensor_calibration_samples)}/{SENSOR_CALIBRATION_COUNT})", end='\r')
                else:
                    queue_delay_cpp_ms = (last_send_ts_in_batch - last_ts_in_batch) * 1000
                    net_delay_ms = (net_plus_offset_s - self.sensor_clock_offset_s) * 1000
                    corrected_total_delay_ms = queue_delay_cpp_ms + net_delay_ms
                    
                    batch_count_sec += 1
                    packet_count_sec += num_packets_in_batch
                    latency_samples_sec.append(corrected_total_delay_ms)
            
            current_time = time.time()
            if current_time - last_log_time >= 1.0:
                current_status = {}
                if self.sensor_clock_offset_s is not None:
                    avg_lat = np.mean(latency_samples_sec) if latency_samples_sec else 0.0
                    with self.lock:
                        current_buffer_size = len(self.sensor_save_buffer)
                    
                    current_status = {
                        "avg_latency": avg_lat,
                        "batch_count": batch_count_sec,
                        "packet_count": packet_count_sec,
                        "buffer_size": current_buffer_size,
                        "last_recv_wall": current_time if batch_count_sec > 0 else self.sensor_latest_status.get("last_recv_wall", 0.0)
                    }
                    latency_samples_sec.clear()
                    batch_count_sec = 0
                    packet_count_sec = 0
                else:
                    current_status = self.sensor_latest_status.copy()
                    current_status["last_recv_wall"] = current_time
                
                with self.lock:
                    self.sensor_latest_status = current_status
                
                last_log_time = current_time
        
        sock.close()
        print("🛑 SensorReceiver stopped")

# ===================== 메인 =====================
def main():
    print("\n" + "="*80)
    print("🚀 Ultra-Fast Receiver (18fps Optimized)")
    print("   - Batch saving (25 frames)")
    print("   - 20 parallel workers")
    print("   - 256MB buffers")
    print("="*80 + "\n")
    
    ctx = zmq.Context.instance()
    
    # 카메라 소켓 (최대 버퍼)
    cam_sock = ctx.socket(zmq.PULL)
    cam_sock.setsockopt(zmq.RCVHWM, 20000)
    cam_sock.setsockopt(zmq.RCVBUF, 256 * 1024 * 1024)  # 256MB
    cam_sock.setsockopt(zmq.RCVTIMEO, -1)
    cam_bind_addr = f"tcp://0.0.0.0:{ZMQ_CAM_PULL_PORT}"
    cam_sock.bind(cam_bind_addr)
    print(f"✅ Camera PULL: {cam_bind_addr} (256MB buffer)")
    
    # 로봇 소켓
    robot_sock = ctx.socket(zmq.SUB)
    robot_sock.setsockopt(zmq.RCVHWM, 100)
    robot_connect_addr = f"tcp://{ZMQ_ROBOT_PUB_ADDRESS}:{ZMQ_ROBOT_PUB_PORT}"
    robot_sock.connect(robot_connect_addr)
    robot_sock.subscribe(ZMQ_ROBOT_TOPIC)
    print(f"✅ Robot SUB: {robot_connect_addr}")
    
    # Ultra-Fast 이미지 저장 (20 워커, 15000 큐)
    writer = UltraFastImageWriter(num_workers=20, max_queue=15000)
    
    # 수신 스레드 시작
    cam_receiver = CameraReceiver(cam_sock, writer)
    robot_receiver = RobotReceiver(robot_sock)
    sensor_receiver = SensorReceiver()
    
    cam_receiver.start()
    robot_receiver.start()
    sensor_receiver.start()
    
    print("\n" + "="*80)
    print("🚀 ALL RECEIVER THREADS STARTED")
    print("="*80 + "\n")
    
    last_status_print = time.time()
    last_fps_print = time.time()
    
    try:
        while not stop_event.is_set():
            now_wall = time.time()
            
            # 상태 모니터링
            if now_wall - last_status_print >= STATUS_PERIOD:
                cam_stats = cam_receiver.get_stats()
                robot_stats = robot_receiver.get_stats()
                sensor_stats = sensor_receiver.get_stats()
                
                print(f"\n--- Status ({datetime.now().strftime('%H:%M:%S')}) ---")
                
                # 카메라
                print("📷 Cameras:")
                if not cam_stats['cam_latest_recv']:
                    print("  Waiting...")
                else:
                    for k in sorted(cam_stats['cam_latest_recv'].keys()):
                        data = cam_stats['cam_latest_recv'][k]
                        d = cam_stats['cam_delta_dict'].get(k, 0.0)
                        fps = (1.0 / d) if d > 0 else 0.0
                        net_ms = data['net_delay'] * 1000.0
                        total_ms = data['total_delay'] * 1000.0
                        
                        # 네거티브 지연 표시 수정
                        net_display = f"{net_ms:6.1f}" if net_ms >= 0 else f"~{abs(net_ms):5.1f}"
                        net_flag = "⚠️" if net_ms > WARN_NET_MS else ""
                        total_flag = "⚠️" if total_ms > WARN_TOTAL_MS else ""
                        stall = "⛔" if (now_wall - cam_stats['cam_last_recv_wall'].get(k, 0)) >= STALL_SEC else ""
                        
                        print(f"  {k:<25} | Δnet:{net_display}ms{net_flag} | "
                              f"Δtotal:{total_ms:6.1f}ms{total_flag} | "
                              f"{fps:5.2f}fps {stall}")
                
                # 로봇
                print("🤖 Robot:")
                if robot_stats['robot_latest_state']:
                    r = robot_stats['robot_latest_state']
                    net_ms = r['net_delay'] * 1000.0
                    total_ms = r['total_delay'] * 1000.0
                    stall = "⛔" if (now_wall - robot_stats['robot_last_recv_wall']) >= STALL_SEC else ""
                    print(f"  Δnet:{net_ms:6.1f}ms | Δtotal:{total_ms:6.1f}ms | "
                          f"J1:{r['joints'][0]:<6.1f} {stall}")
                else:
                    print("  Waiting...")
                
                # 센서
                print("🔬 Sensor:")
                s_lat = sensor_stats.get('avg_latency', 0.0)
                s_buf = sensor_stats.get('buffer_size', 0)
                s_last = sensor_stats.get('last_recv_wall', 0.0)
                s_stall = "⛔" if (now_wall - s_last) >= STALL_SEC else ""
                
                if sensor_receiver.sensor_clock_offset_s is not None:
                    print(f"  Avg Lat: {s_lat:6.1f}ms | Buffer: {s_buf:<6d} {s_stall}")
                else:
                    print(f"  Calibrating... {s_stall}")
                
                print(f"💾 Writer Queue: {writer.q.qsize()}")
                last_status_print = now_wall
            
            # FPS 출력
            if now_wall - last_fps_print >= FPS_PERIOD:
                elapsed = now_wall - last_fps_print
                
                cam_stats = cam_receiver.get_stats()
                robot_stats = robot_receiver.get_stats()
                
                if cam_stats['cam_cnt']:
                    line = " | ".join([f"{k}:{cam_stats['cam_cnt'][k]/elapsed:.1f}fps"
                                     for k in sorted(cam_stats['cam_cnt'])])
                    print(f"📊 Camera Avg FPS: {line}")
                
                robot_cnt = robot_stats['robot_cnt']
                if robot_cnt > 0:
                    print(f"📊 Robot: {robot_cnt/elapsed:.1f} msg/s")
                
                # 카운터 리셋
                cam_receiver.reset_counters()
                robot_receiver.reset_counter()
                
                last_fps_print = now_wall
            
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C in main loop")
    
    finally:
        print("\n--- Shutdown Sequence ---")
        stop_event.set()
        
        # 스레드 종료 대기
        print("⏳ Waiting for threads...")
        cam_receiver.join(timeout=5.0)
        if cam_receiver.is_alive():
            print("[WARN] Camera thread did not exit cleanly")
        else:
            print("✅ Camera thread stopped")
        
        robot_receiver.join(timeout=5.0)
        if robot_receiver.is_alive():
            print("[WARN] Robot thread did not exit cleanly")
        else:
            print("✅ Robot thread stopped")
        
        sensor_receiver.join(timeout=5.0)
        if sensor_receiver.is_alive():
            print("[WARN] Sensor thread did not exit cleanly")
        else:
            print("✅ Sensor thread stopped")
        
        # 데이터 저장
        print("\n💾 Saving data...")
        
        # 로봇 데이터
        robot_data = robot_receiver.get_data_buffer()
        if robot_data:
            save_robot_data_to_csv(robot_data, ROBOT_CSV_PATH)
        
        # 센서 데이터
        sensor_data = sensor_receiver.get_data_buffer()
        if sensor_data:
            save_sensor_data_to_npz(sensor_data, SENSOR_NPZ_BASE_FILENAME)
        
        # 이미지 저장 종료
        writer.stop()
        
        # 최종 통계
        print("\n--- Final Summary ---")
        
        cam_stats = cam_receiver.get_stats()
        robot_stats = robot_receiver.get_stats()
        sensor_stats = sensor_receiver.get_stats()
        
        print("📷 Cameras:")
        for k in sorted(cam_stats['cam_latest_recv'].keys()):
            data = cam_stats['cam_latest_recv'][k]
            fail = cam_stats['cam_fail_count'].get(k, 0)
            print(f"  {k:<25} | last_cap:{data['capture_time']:.3f} | "
                  f"last_recv:{data['recv_time']:.3f} | fail:{fail}")
        
        print("🤖 Robot:")
        if robot_stats['robot_latest_state']:
            r = robot_stats['robot_latest_state']
            print(f"  last_recv:{r['recv_time']:.3f} | J1:{r['joints'][0]:.1f}")
        else:
            print("  No data received")
        
        print("🔬 Sensor:")
        print(f"  Avg Latency: {sensor_stats.get('avg_latency', 0):.1f}ms | "
              f"Buffer Size: {sensor_stats.get('buffer_size', 0)}")
        
        # ZMQ 정리
        print("\n🧹 Cleaning up ZMQ...")
        try:
            cam_sock.close()
            robot_sock.close()
            ctx.term()
        except Exception as e:
            print(f"[WARN] ZMQ cleanup error: {e}")
        
        # GC
        print("♻️ Running Garbage Collection...")
        gc.collect()
        
        print("\n✅✅ Receiver shutdown complete.")

if __name__ == "__main__":
    main()