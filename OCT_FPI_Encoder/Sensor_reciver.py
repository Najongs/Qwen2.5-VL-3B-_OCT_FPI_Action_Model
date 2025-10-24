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

TOTAL_PACKET_SIZE = 8 + 8 + 4 + (4 * NXZRt)
PACKET_FORMAT = f'<d d f {NXZRt}f'

# 상수가 올바른지 Python 자체적으로 검증
assert struct.calcsize(PACKET_FORMAT) == TOTAL_PACKET_SIZE, \
    f"Struct 크기 불일치! C++: {TOTAL_PACKET_SIZE}, Python: {struct.calcsize(PACKET_FORMAT)}"

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
def unpack_batch(buffer_data, num_packets):
    """
    C++ DataPacket (4120 바이트) * num_packets 개수만큼의
    버퍼 데이터를 파싱하여 레코드 리스트로 반환합니다.
    """
    records = []
    
    # 예상 버퍼 크기와 실제 버퍼 크기가 일치하는지 확인
    expected_size = num_packets * TOTAL_PACKET_SIZE
    if len(buffer_data) != expected_size:
        print(f"[ERROR] unpack_batch: 크기 불일치! "
              f"예상: {expected_size}B, 실제: {len(buffer_data)}B")
        # 크기가 안 맞으면 데이터가 깨지므로 빈 리스트 반환
        return [] 

    try:
        # iter_unpack을 사용하여 버퍼를 4120 바이트 단위로 순회
        for unpacked_data in struct.iter_unpack(PACKET_FORMAT, buffer_data):
            # unpacked_data[0]: ts (double)
            # unpacked_data[1]: send_ts (double)
            # unpacked_data[2]: force (float)
            # unpacked_data[3:]: aline_data (float * 1025 튜플)
            
            records.append({
                "timestamp": unpacked_data[0],
                "send_timestamp": unpacked_data[1],
                "force": unpacked_data[2],
                # 튜플(unpacked_data[3:])을 리스트로 변환 (필요시)
                "aline": list(unpacked_data[3:]) 
            })
            
    except struct.error as e:
        print(f"[ERROR] struct.iter_unpack 실패: {e}. 포맷/데이터 손상 확인 필요.")
        # 에러 발생 시 빈 리스트 반환
        return []

    # 헤더에 명시된 패킷 수와 실제 파싱된 수가 다르면 경고
    if len(records) != num_packets:
        print(f"[WARNING] 패킷 수 불일치! 헤더: {num_packets}개, 실제 파싱: {len(records)}개")

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

    # 1초 요약 로그 변수
    last_log_time = time.time()
    batch_count_sec = 0
    packet_count_sec = 0
    latency_samples_sec = []

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

        # ▼▼▼ [수정] 헤더/페이로드 순서 꼬임 방지 로직 ▼▼▼
        
        # 1. 4바이트 헤더(패킷 개수) 수신 시
        if len(data) == 4:
            # 만약 이전에 처리 못한 페이로드가 버퍼에 남아있다면,
            # (즉, 헤더가 연속 두 번 들어온 이례적인 상황)
            # 기존 버퍼는 비우고 새로 시작
            if expected_payload_size > 0 or pending_num_packets > 0:
                 print(f"[WARNING] 새 헤더 수신. 이전 버퍼( {len(buffer)}B )를 비웁니다.")
                 buffer.clear()

            pending_num_packets = struct.unpack('<I', data)[0]
            if pending_num_packets > 0:
                expected_payload_size = pending_num_packets * TOTAL_PACKET_SIZE
            else:
                # 0개짜리 헤더가 오면 무시
                expected_payload_size = 0
                pending_num_packets = 0
            
            # 중요: 헤더 수신 시 buffer.clear()를 하지 않음
            # (헤더보다 페이로드가 먼저 도착한 경우를 대비)
            
            # print(f"헤더 수신: {pending_num_packets}개, {expected_payload_size}B 대기") # (디버그용)
            
            # 다음 루프로 가서 페이로드 수신 대기
            # (단, 이미 버퍼에 데이터가 차있을 수 있으므로 continue 안 함)
            pass

        # 2. 페이로드 데이터 수신 시
        elif len(data) > 4:
             buffer.extend(data)
             # print(f"페이로드 수신: {len(data)}B / 누적 {len(buffer)}B") # (디버그용)
        
        # 3. 헤더/페이로드 무관하게 버퍼 체크
        # (expected_payload_size가 0보다 커야 함 = 헤더를 1번 이상 받음)
        if expected_payload_size > 0 and len(buffer) >= expected_payload_size:
            
            # (디버그용) 정확히 맞지 않으면 경고
            if len(buffer) > expected_payload_size:
                 print(f"[WARNING] 버퍼가 예상보다 큼! {len(buffer)}B > {expected_payload_size}B. "
                       f"초과분 {len(buffer) - expected_payload_size}B 남김.")

            recv_time = time.time() # T_recv
            
            # 정확히 예상 크기만큼만 잘라서 처리
            payload_to_process = buffer[:expected_payload_size]
            # 처리한 부분은 버퍼에서 제거
            buffer = buffer[expected_payload_size:]
            
            try:
                # ▼▼▼ [수정] 새 unpack_batch 함수 호출 ▼▼▼
                records = unpack_batch(payload_to_process, pending_num_packets)
            except Exception as e:
                print(f"[ERROR] 언팩 실패: {e}")
                # 상태 초기화
                buffer.clear()
                pending_num_packets = 0
                expected_payload_size = 0
                continue
            
            # 언팩 실패 시(데이터 손상) records가 비어있을 수 있음
            if not records:
                 print("[ERROR] 언팩 결과 데이터가 없습니다. 이 배치를 건너뜁니다.")
                 # 상태 초기화
                 pending_num_packets = 0
                 expected_payload_size = 0
                 continue
                 
            # ▲▲▲ [수정] ▲▲▲

            # (이하 로직은 기존과 동일)
            
            # 마지막 패킷의 타임스탬프 (오차 계산용)
            last_ts = records[-1]["timestamp"]
            last_send_ts = records[-1]["send_timestamp"]
            
            with lock:
                continuous_sensor_data.extend(records)
            with save_lock:
                save_buffer.extend(records)
            
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

            # 버퍼 처리가 끝났으므로 상태 초기화
            # (주의: buffer.clear()가 아님! 위에서 이미 잘라냈음)
            pending_num_packets = 0
            expected_payload_size = 0
        
        # [1초마다 요약 로그 출력]
        current_time = time.time()
        if CLOCK_OFFSET_SECONDS is not None and (current_time - last_log_time >= 1.0):
            if batch_count_sec > 0:
                avg_latency = np.mean(latency_samples_sec)
                print(f"📡 1초간 수신: {batch_count_sec}개 배치 ({packet_count_sec}개 패킷) | "
                      f"평균 총지연: {avg_latency:.1f}ms | "
                      f"버퍼: {len(continuous_sensor_data)}")
                
                # 카운터 초기화
                latency_samples_sec.clear()
                batch_count_sec = 0
                packet_count_sec = 0
            
            last_log_time = current_time

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