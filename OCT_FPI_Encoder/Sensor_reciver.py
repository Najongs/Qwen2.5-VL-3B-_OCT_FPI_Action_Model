import socket
import struct
import time
import datetime
import os
from collections import deque
import threading
import numpy as np
# (Optional: for saving the M-mode image for debugging)
# import matplotlib.pyplot as plt

# --- 설정 (C++과 일치) ---
NXZRt = 1025
HOST = '0.0.0.0'
PORT = 9999

# --- 리샘플링 설정 ---
TIMEOUT_SEC = 2.0
MODEL_INPUT_DURATION_SEC = 0.5 # 모델 입력의 시간 길이 (0.5초)
MODEL_INPUT_NUM_SAMPLES = 300 # 모델 입력의 샘플 개수 (300개)

# --- 데이터 구조체 계산 ---
PACKET_HEADER_FORMAT = '<df'
PACKET_HEADER_SIZE = struct.calcsize(PACKET_HEADER_FORMAT)
ALINE_FORMAT = f'<{NXZRt}f'
ALINE_SIZE = struct.calcsize(ALINE_FORMAT)
TOTAL_PACKET_SIZE = PACKET_HEADER_SIZE + ALINE_SIZE

# --- 데이터 버퍼 ---
continuous_sensor_data = deque() # 모든 센서 데이터를 연속적으로 저장하는 버퍼
lock = threading.Lock()

def model_input_generator_thread():
    """
    주기적으로 버퍼를 확인하여 모델 입력을 생성합니다.
    """
    while True:
        time.sleep(MODEL_INPUT_DURATION_SEC) # 0.5초마다 실행
        
        with lock:
            if len(continuous_sensor_data) < 2:
                continue

            # 가장 최근 데이터의 타임스탬프를 기준으로 0.5초 윈도우 설정
            t_end = continuous_sensor_data[-1]['timestamp']
            t_start = t_end - MODEL_INPUT_DURATION_SEC

            # 1. 0.5초 윈도우에 해당하는 데이터 추출
            window_data = [p for p in continuous_sensor_data if t_start <= p['timestamp'] <= t_end]
            
            # 데이터가 2개 미만이면 보간 불가
            if len(window_data) < 2:
                continue

            # 2. 리샘플링 준비
            original_timestamps = np.array([p['timestamp'] for p in window_data])
            original_values = np.array([np.concatenate(([p['force']], p['aline'])) for p in window_data])

            # 새로 만들, 균일한 간격의 타임스탬프
            resampled_timestamps = np.linspace(t_start, t_end, MODEL_INPUT_NUM_SAMPLES)
            
            # 3. 보간(Interpolation) 실행
            resampled_values = np.zeros((MODEL_INPUT_NUM_SAMPLES, NXZRt + 1))
            for i in range(NXZRt + 1):
                resampled_values[:, i] = np.interp(resampled_timestamps, original_timestamps, original_values[:, i])

            # 4. 최종 데이터 분리 및 처리
            force_data = resampled_values[:, 0]
            mmode_image_data = resampled_values[:, 1:]
            
            print("\n✅ 최종 데이터 생성 성공!")
            print(f"  - 센서 데이터 (시작 TS: {t_start:.3f}, 길이: {MODEL_INPUT_DURATION_SEC}초, 샘플 수: {MODEL_INPUT_NUM_SAMPLES}개)")
            print(f"  - Force 데이터 shape: {force_data.shape}")
            print(f"  - M-mode 이미지 shape: {mmode_image_data.shape}")

            # >> 여기서 AI 모델에 데이터를 전달합니다 <<
            # model.predict(force_data, mmode_image_data)

            # 5. 오래된 데이터 버퍼에서 정리
            cutoff_time = t_start - TIMEOUT_SEC
            while continuous_sensor_data and continuous_sensor_data[0]['timestamp'] < cutoff_time:
                continuous_sensor_data.popleft()


def sensor_receiver_thread():
    """소켓으로 센서 데이터를 수신하여 연속 버퍼에 추가"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"센서 수신 스레드: {PORT}에서 C++ 접속 대기 중...")
        conn, addr = s.accept()
        with conn:
            print(f"센서 수신 스레드: {addr} 에서 연결됨.")
            while True:
                try:
                    # 1. 헤더 수신: 이번 배치에 몇 개의 패킷이 있는지 (4바이트)
                    header_data = recv_all(conn, 4)
                    if not header_data:
                        print("클라이언트 연결 끊어짐 (헤더 수신 실패).")
                        break
                    
                    num_packets = struct.unpack('<I', header_data)[0]
                    
                    # 2. 본문 수신: (패킷 개수 * 패킷 크기) 만큼의 데이터 수신
                    total_data_size = num_packets * TOTAL_PACKET_SIZE
                    batch_data_bytes = recv_all(conn, total_data_size)
                    
                    if not batch_data_bytes:
                        print("클라이언트 연결 끊어짐 (데이터 수신 실패).")
                        break

                    # 받은 배치를 연속 버퍼에 추가
                    for i in range(num_packets):
                        offset = i * TOTAL_PACKET_SIZE
                        header = batch_data_bytes[offset : offset + PACKET_HEADER_SIZE]
                        ts, force = struct.unpack(PACKET_HEADER_FORMAT, header)
                        aline_part = batch_data_bytes[offset + PACKET_HEADER_SIZE : offset + TOTAL_PACKET_SIZE]
                        aline = struct.unpack(ALINE_FORMAT, aline_part)
                        
                        with lock:
                            continuous_sensor_data.append({'timestamp': ts, 'force': force, 'aline': aline})
                    
                    latest_ts = continuous_sensor_data[-1]['timestamp']
                    print(f"📦 센서 배치 수신 완료 ({num_packets}개). 버퍼 크기: {len(continuous_sensor_data)}, 최근 TS: {latest_ts:.3f}")

                except (ConnectionResetError, BrokenPipeError):
                    print("클라이언트가 연결을 종료했습니다.")
                    break
                except Exception as e:
                    print(f"수신 중 오류 발생: {e}")
                    break

    print("센서 수신 스레드 종료.")


def recv_all(conn, n_bytes):
    """소켓에서 정확히 n_bytes 만큼의 데이터를 수신하는 헬퍼 함수"""
    data = bytearray()
    while len(data) < n_bytes:
        packet = conn.recv(n_bytes - len(data))
        if not packet: return None
        data.extend(packet)
    return data

if __name__ == "__main__":
    # 두 개의 스레드를 시작
    sensor_thread = threading.Thread(target=sensor_receiver_thread)
    model_input_thread = threading.Thread(target=model_input_generator_thread, daemon=True)

    sensor_thread.start()
    model_input_thread.start()

    sensor_thread.join()
    print("메인 프로그램 종료.")

