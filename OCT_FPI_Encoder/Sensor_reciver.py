import socket
import struct
import threading
import time
import signal
from collections import deque
import numpy as np

# =========================
# 설정 (C++과 일치)
# =========================
NXZRt = 1025
UDP_PORT = 9999
UDP_IP = "0.0.0.0"
BUFFER_SIZE = 65535 # (UDP 최대 크기)

# =========================
# 리샘플링 / 모델 입력
# =========================
TIMEOUT_SEC = 2.0
MODEL_INPUT_DURATION_SEC = 1.0
MODEL_INPUT_CHECK_INTERVAL = 1.0 # 0.2에서 1.0으로 수정
MODEL_INPUT_NUM_SAMPLES = 650

# =========================
# 프로토콜 사이즈 (4120B)
# =========================
PACKET_HEADER_FORMAT = '<ddf'
PACKET_HEADER_SIZE = struct.calcsize(PACKET_HEADER_FORMAT)  # 20B
ALINE_FORMAT = f'<{NXZRt}f'
ALINE_SIZE = struct.calcsize(ALINE_FORMAT)                  # 4100B
TOTAL_PACKET_SIZE = PACKET_HEADER_SIZE + ALINE_SIZE         # 4120B

# =========================
# 공유 버퍼 / 상태
# =========================
continuous_sensor_data = deque(maxlen=20000) # 모델 입력용 버퍼
lock = threading.Lock()
stop_event = threading.Event()

last_corrected_total_delay = 0.0
last_recv_ts = 0.0
CLOCK_OFFSET_SECONDS = None
CALIBRATION_SAMPLES = []
CALIBRATION_COUNT = 50

# ▼▼▼ [추가] 데이터 저장을 위한 전역 변수 ▼▼▼
save_buffer = []
save_lock = threading.Lock()
# ▲▲▲ [추가] ▲▲▲


# =========================
# 시그널 핸들러
# =========================
def handle_sigint(sig, frame):
    print("\n🛑 Ctrl+C 감지 — 안전 종료 중...")
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)


# =========================
# 데이터 언팩 (4120B)
# =========================
def unpack_batch(payload_bytes: bytes, num_packets: int):
    records = []
    mv = memoryview(payload_bytes)
    offset = 0
    for _ in range(num_packets):
        header = mv[offset:offset + PACKET_HEADER_SIZE]
        ts, send_ts, force = struct.unpack(PACKET_HEADER_FORMAT, header)
        offset += PACKET_HEADER_SIZE

        aline_bytes = mv[offset:offset + ALINE_SIZE]
        aline = np.frombuffer(aline_bytes, dtype=np.float32).copy()
        offset += ALINE_SIZE
        records.append((ts, send_ts, float(force), aline))
    return records

# =========================
# UDP 수신 스레드 (수정됨 - 1초 요약 로그)
# =========================
def udp_receiver_thread():
    global last_corrected_total_delay, last_recv_ts, CLOCK_OFFSET_SECONDS

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)
    print(f"✅ UDP 리시버 시작 (포트 {UDP_PORT}) - [시계 오차 자동 보정 모드]")
    print(f"⏳ 최초 {CALIBRATION_COUNT}개 배치로 C++/Python 간 시계 오차를 보정합니다...")
    buffer = bytearray()
    expected_payload_size = 0
    pending_num_packets = 0

    # ▼▼▼ 1초 요약 로그를 위한 변수 ▼▼▼
    last_log_time = time.time()
    batch_count_sec = 0
    packet_count_sec = 0
    latency_samples_sec = []
    # ▲▲▲

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
        except socket.timeout:
            continue
        except Exception as e:
            if stop_event.is_set():
                break
            print(f"[UDP] 수신 오류: {e}")
            continue

        if len(data) == 4:
            pending_num_packets = struct.unpack('<I', data)[0]
            expected_payload_size = pending_num_packets * TOTAL_PACKET_SIZE
            buffer.clear()
            continue

        buffer.extend(data)
        if len(buffer) >= expected_payload_size > 0:
            recv_time = time.time() # T_recv
            
            try:
                records = unpack_batch(buffer[:expected_payload_size], pending_num_packets)
            except Exception as e:
                print(f"[ERROR] 언팩 실패: {e}")
                buffer.clear()
                continue

            last_ts, last_send_ts = 0.0, 0.0
            
            dict_records = []
            for ts, send_ts, force, aline in records:
                last_ts, last_send_ts = ts, send_ts
                dict_records.append({
                    "timestamp": ts,
                    "send_timestamp": send_ts,
                    "force": force,
                    "aline": aline
                })

            with lock:
                continuous_sensor_data.extend(dict_records)
            with save_lock:
                save_buffer.extend(dict_records)
            
            last_recv_ts = recv_time

            # --- [시계 오차 보정 로직] ---
            queue_delay_cpp_ms = (last_send_ts - last_ts) * 1000
            net_plus_offset_s = recv_time - last_send_ts

            if CLOCK_OFFSET_SECONDS is None:
                CALIBRATION_SAMPLES.append(net_plus_offset_s)
                if len(CALIBRATION_SAMPLES) >= CALIBRATION_COUNT:
                    CLOCK_OFFSET_SECONDS = np.mean(CALIBRATION_SAMPLES)
                    print("\n" + "="*80)
                    print(f"✅ 보정 완료! 계산된 시계 오차: {CLOCK_OFFSET_SECONDS * 1000:.1f} ms")
                    print("이제부터 실제 성능 지연을 표시합니다.")
                    print("="*80 + "\n")
                else:
                     print(f"⏳ 보정 중... ({len(CALIBRATION_SAMPLES)}/{CALIBRATION_COUNT})", end='\r')
            else:
                net_delay_ms = (net_plus_offset_s - CLOCK_OFFSET_SECONDS) * 1000
                corrected_total_delay_ms = queue_delay_cpp_ms + net_delay_ms
                last_corrected_total_delay = corrected_total_delay_ms / 1000.0

                # 1초간 데이터 수집
                batch_count_sec += 1
                packet_count_sec += pending_num_packets
                latency_samples_sec.append(corrected_total_delay_ms)

            buffer.clear()
            pending_num_packets = 0
            expected_payload_size = 0
        
        # ▼▼▼ [수정] 1초마다 요약 로그 출력 ▼▼▼
        current_time = time.time()
        if CLOCK_OFFSET_SECONDS is not None and (current_time - last_log_time >= 1.0): # 5.0 -> 1.0
            if batch_count_sec > 0:
                avg_latency = np.mean(latency_samples_sec)
                # "5초간" -> "1초간"
                print(f"📡 1초간 수신: {batch_count_sec}개 배치 ({packet_count_sec}개 패킷) | "
                      f"평균 총지연: {avg_latency:.1f}ms | "
                      f"버퍼: {len(continuous_sensor_data)}")
                
                # 카운터 초기화
                latency_samples_sec.clear()
                batch_count_sec = 0
                packet_count_sec = 0
            
            last_log_time = current_time
        # ▲▲▲ [수정] ▲▲▲

    sock.close()
    print("UDP 수신 스레드 종료.")

# ▼▼▼ [수정] 데이터 저장을 위한 함수 (파일명에 타임스탬프 포함) ▼▼▼
def save_data_to_npz():
    """
    프로그램 종료 시 save_buffer에 누적된 데이터를 .npz 파일로 저장합니다.
    파일명은 첫 번째 데이터의 타임스탬프를 포함합니다.
    """
    
    filename = "saved_data_empty.npz" # 기본 파일명 (데이터가 없는 경우)
    
    with save_lock:
        if not save_buffer:
            print("저장할 데이터가 없습니다.")
            return

        try:
            # 1. 첫 번째 타임스탬프를 가져와 파일명 생성
            first_ts_int = int(save_buffer[0]['timestamp'])
            filename = f"saved_data_{first_ts_int}.npz"
            
            print(f"\n💾 저장 중... {len(save_buffer)}개의 레코드를 {filename}에 저장합니다...")

            # 2. 리스트 딕셔너리를 Numpy 배열로 변환
            timestamps = np.array([d['timestamp'] for d in save_buffer], dtype=np.float64)
            send_timestamps = np.array([d['send_timestamp'] for d in save_buffer], dtype=np.float64)
            forces = np.array([d['force'] for d in save_buffer], dtype=np.float32)
            alines = np.array([d['aline'] for d in save_buffer], dtype=np.float32)
            
            # 3. 압축하여 .npz 파일로 저장
            np.savez(filename, 
                    timestamps=timestamps,
                    send_timestamps=send_timestamps,
                    forces=forces,
                    alines=alines)
            
            print(f"✅ 저장 완료! ({filename})")
            print(f"  - timestamps: {timestamps.shape}")
            print(f"  - send_timestamps: {send_timestamps.shape}")
            print(f"  - forces: {forces.shape}")
            print(f"  - alines: {alines.shape}")

        except Exception as e:
            print(f"[ERROR] 파일 저장 실패 ({filename}): {e}")
# ▲▲▲ [수정] ▲▲▲


# =========================
# 모델 입력 생성(슬라이딩 윈도우)
# =========================
def model_input_generator_thread():
    global last_corrected_total_delay, last_recv_ts

    while not stop_event.is_set():
        time.sleep(MODEL_INPUT_CHECK_INTERVAL)

        with lock:
            if len(continuous_sensor_data) < 2:
                continue
            t_end = continuous_sensor_data[-1]['timestamp']
            t_start = t_end - MODEL_INPUT_DURATION_SEC
            window_data = [p for p in continuous_sensor_data if t_start <= p['timestamp'] <= t_end]
            if len(window_data) < 2:
                continue

        t_proc_start = time.time()
        original_timestamps = np.array([p['timestamp'] for p in window_data])
        original_values = np.array([np.concatenate(([p['force']], p['aline'])) for p in window_data])
        
        sort_indices = np.argsort(original_timestamps)
        original_timestamps, original_values = original_timestamps[sort_indices], original_values[sort_indices]

        resampled_timestamps = np.linspace(t_start, t_end, MODEL_INPUT_NUM_SAMPLES)
        resampled_values = np.empty((MODEL_INPUT_NUM_SAMPLES, NXZRt + 1), dtype=np.float32)

        for i in range(NXZRt + 1):
            resampled_values[:, i] = np.interp(resampled_timestamps, original_timestamps, original_values[:, i])

        force_data, mmode_image_data = resampled_values[:, 0], resampled_values[:, 1:]

        t_proc_end = time.time()
        processing_delay = t_proc_end - t_proc_start
        pipeline_delay = t_proc_end - last_recv_ts if last_recv_ts > 0 else 0

        print(f"\n✅ 모델 입력 생성 성공")
        print(f"  ⏱ 구간: {t_start:.3f} ~ {t_end:.3f} | 길이 1.0s")
        print(f"  Force shape: {force_data.shape}, M-mode shape: {mmode_image_data.shape}")
        print(f"  📶 실제 총 지연: {last_corrected_total_delay*1000:.1f} ms | ⚙️ Processing: {processing_delay:.3f}s | "
              f"🕒 Pipeline: {pipeline_delay:.3f}s")

        with lock:
            cutoff_time = t_start - TIMEOUT_SEC
            while continuous_sensor_data and continuous_sensor_data[0]['timestamp'] < cutoff_time:
                continuous_sensor_data.popleft()


# =========================
# 실행부
# =========================
if __name__ == "__main__":
    udp_thread = threading.Thread(target=udp_receiver_thread, daemon=False)
    model_thread = threading.Thread(target=model_input_generator_thread, daemon=True)

    udp_thread.start()
    model_thread.start()

    try:
        udp_thread.join()
    finally:
        stop_event.set() 
        
        # ▼▼▼ [수정] 파일명 인자 없이 저장 함수 호출 ▼▼▼
        save_data_to_npz()
        # ▲▲▲ [수정] ▲▲▲
        
        print("✅ 메인 프로그램 종료.")