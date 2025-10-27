#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카메라 Sender 2Hz 버전
- 초당 2프레임 전송 (0.5초 간격)
- 처리량 2배 증가 대비 최적화
"""

import os, time, json, cv2, zmq, numpy as np, threading
from queue import Queue, Empty
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent
from collections import deque
import depthai as dai
import pyzed.sl as sl
import signal

# ===================== 설정 =====================
SERVER_IP = "10.130.4.79"
SERVER_PORT = 5555

# 🔥 2Hz 설정 (핵심!)
CAPTURE_INTERVAL = 0.5  # 0.5초마다 캡처 (2Hz)
PULSE_WIDTH = 0.01  # 10ms 펄스 (더 짧게)

# JPEG 최적화 (2Hz에 맞춰 조정)
JPEG_QUALITY = 70  # 75→70 (처리량 증가 대비)
JPEG_OPTIMIZE = False
JPEG_PROGRESSIVE = False

SEND_ZED_RIGHT = True  # False → True (Right 카메라 활성화!)

# ZMQ 최적화 (2Hz 대응)
ZMQ_IO_THREADS = 8  # 6→8 (처리량 증가)
CAMERA_SNDHWM = 8000  # 5000→8000
SNDBUF_SIZE = 64 * 1024 * 1024  # 32MB→64MB

# 인코딩 병렬화 증가 (2Hz 대응)
NUM_ENCODER_PROCESSES = 8  # 6→8 (초당 10프레임 처리 필요)
ENCODING_QUEUE_SIZE = 1000  # 800→1000
BATCH_SEND_SIZE = 2  # 3→2 (더 빠른 전송)
BATCH_TIMEOUT = 0.015  # 0.02→0.015초

# 프레임 전처리
RESIZE_BEFORE_ENCODE = True
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720

# ===================== 전역 플래그 =====================
stop_flag = threading.Event()
encoder_stop_flag = MPEvent()

def handle_sigint(sig, frame):
    print("\n🛑 Ctrl+C detected — Shutting down...")
    stop_flag.set()
    encoder_stop_flag.set()

signal.signal(signal.SIGINT, handle_sigint)

# ===================== 고정밀 트리거 (2Hz) =====================
class HighFreqTrigger(threading.Thread):
    """
    2Hz 고정밀 트리거
    - 정확한 0.5초 간격 유지
    - 드리프트 자동 보정
    """
    def __init__(self, interval=0.5, pulse_width=0.01):
        super().__init__(daemon=True)
        self.interval = float(interval)
        self.pulse_width = float(pulse_width)
        self.event = threading.Event()
        self.frame_count = 0
        
        # 타이밍 통계
        self.last_trigger_times = deque(maxlen=20)
        
        print(f"⏱ HighFreqTrigger: {1/interval:.1f} Hz ({interval*1000:.0f}ms interval)")
    
    def run(self):
        print("⏱ HighFreqTrigger started")
        next_trigger = time.time() + self.interval
        
        while not stop_flag.is_set():
            now = time.time()
            
            if now >= next_trigger:
                trigger_time = time.time()
                self.event.set()
                self.frame_count += 1
                
                # 타이밍 기록
                self.last_trigger_times.append(trigger_time)
                
                # 펄스 유지
                time.sleep(self.pulse_width)
                self.event.clear()
                
                # 다음 트리거 계산
                next_trigger += self.interval
                
                # 드리프트 보정
                if now - next_trigger > self.interval:
                    next_trigger = now + self.interval
                    print(f"⚠️ Trigger drift corrected at frame {self.frame_count}")
            else:
                # 정밀 대기
                sleep_time = next_trigger - now
                if sleep_time > 0.001:
                    time.sleep(sleep_time * 0.7)  # 70% 대기
                else:
                    time.sleep(0.0001)  # Busy-wait
        
        # 통계 출력
        if len(self.last_trigger_times) >= 2:
            intervals = np.diff(list(self.last_trigger_times))
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            print(f"📊 Trigger Stats: avg={avg_interval*1000:.1f}ms, "
                  f"std={std_interval*1000:.2f}ms, frames={self.frame_count}")
        
        print(f"🛑 HighFreqTrigger stopped (triggers: {self.frame_count})")

# ===================== 고속 JPEG 인코더 =====================
def fast_jpeg_encoder_process(input_queue, output_queue, process_id, quality, resize_enabled):
    """최적화 JPEG 인코더"""
    print(f"🔧 FastEncoder-{process_id} started (Q={quality}, resize={resize_enabled})")
    
    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY), int(quality),
        int(cv2.IMWRITE_JPEG_OPTIMIZE), 0,
        int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0
    ]
    
    encoded_count = 0
    total_encode_time = 0.0
    
    while not encoder_stop_flag.is_set():
        try:
            item = input_queue.get(timeout=0.1)
            if item is None:
                break
            
            cam_name, frame, timestamp = item
            t_start = time.time()
            
            # 리사이즈
            if resize_enabled and frame.shape[1] > RESIZE_WIDTH:
                frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT), 
                                 interpolation=cv2.INTER_AREA)
            
            # JPEG 인코딩
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            
            encode_time = (time.time() - t_start) * 1000
            total_encode_time += encode_time
            
            if ok:
                output_queue.put((cam_name, buf.tobytes(), timestamp, buf.nbytes, encode_time))
                encoded_count += 1
        
        except Empty:
            continue
        except Exception as e:
            print(f"[Encoder-{process_id}] Error: {e}")
    
    avg_encode_time = total_encode_time / encoded_count if encoded_count > 0 else 0
    print(f"🛑 Encoder-{process_id} stopped (frames: {encoded_count}, avg: {avg_encode_time:.1f}ms)")

# ===================== 고속 카메라 전송 (2Hz 대응) =====================
class FastCameraSender(threading.Thread):
    """2Hz 대응 카메라 전송"""
    def __init__(self, ip, port, quality=70, resize=True):
        super().__init__(daemon=True)
        
        # ZMQ 소켓
        self.ctx = zmq.Context.instance()
        self.ctx.setsockopt(zmq.IO_THREADS, ZMQ_IO_THREADS)
        
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, CAMERA_SNDHWM)
        self.sock.setsockopt(zmq.SNDBUF, SNDBUF_SIZE)
        self.sock.setsockopt(zmq.SNDTIMEO, 5)
        self.sock.setsockopt(zmq.IMMEDIATE, 1)
        self.sock.setsockopt(zmq.TCP_KEEPALIVE, 1)  # 연결 유지
        self.sock.connect(f"tcp://{ip}:{port}")
        
        # 인코더 프로세스
        self.encode_input_queue = MPQueue(ENCODING_QUEUE_SIZE)
        self.encode_output_queue = MPQueue(ENCODING_QUEUE_SIZE)
        
        self.encoders = []
        for i in range(NUM_ENCODER_PROCESSES):
            p = Process(
                target=fast_jpeg_encoder_process,
                args=(self.encode_input_queue, self.encode_output_queue, i, quality, resize),
                daemon=True
            )
            p.start()
            self.encoders.append(p)
        
        # 통계 (1초/3초 단위)
        self.stats = {
            'submitted': 0, 'encoded': 0, 'sent': 0,
            'dropped_encode': 0, 'dropped_send': 0,
            'total_encode_time': 0.0,
            'encode_samples': 0,
            'last_log': time.time(),
            'last_sec': time.time(),
            'fps_1sec': 0,
            'fps_3sec': 0
        }
        
        # FPS 계산용
        self.sent_times = deque(maxlen=100)
        
        print(f"📷 FastCameraSender: {NUM_ENCODER_PROCESSES} encoders, Q={quality}")
    
    def submit(self, cam_name, frame, timestamp):
        """카메라 스레드에서 호출"""
        if stop_flag.is_set():
            return
        
        try:
            self.encode_input_queue.put_nowait((cam_name, frame, timestamp))
            self.stats['submitted'] += 1
        except:
            self.stats['dropped_encode'] += 1
    
    def run(self):
        print("📷 FastCameraSender started (2Hz mode)")
        batch = []
        last_batch_send = time.time()
        
        while not stop_flag.is_set():
            # 인코딩 완료 데이터 수집
            try:
                cam_name, jpg_bytes, timestamp, size, encode_time = self.encode_output_queue.get(timeout=0.003)
                self.stats['encoded'] += 1
                self.stats['total_encode_time'] += encode_time
                self.stats['encode_samples'] += 1
                
                meta = {
                    "camera": cam_name,
                    "timestamp": float(timestamp),
                    "send_time": round(time.time(), 3),
                    "size": int(size)
                }
                batch.append((meta, jpg_bytes))
            
            except Empty:
                pass
            
            # 배치 전송
            should_send = (
                len(batch) >= BATCH_SEND_SIZE or
                (len(batch) > 0 and time.time() - last_batch_send > BATCH_TIMEOUT)
            )
            
            if should_send:
                for meta, jpg_bytes in batch:
                    try:
                        self.sock.send_multipart(
                            [json.dumps(meta).encode(), jpg_bytes],
                            flags=zmq.NOBLOCK
                        )
                        self.stats['sent'] += 1
                        self.sent_times.append(time.time())
                    except zmq.Again:
                        self.stats['dropped_send'] += 1
                
                batch.clear()
                last_batch_send = time.time()
            
            self._maybe_log()
        
        # 종료 처리
        if batch:
            for meta, jpg_bytes in batch:
                try:
                    self.sock.send_multipart([json.dumps(meta).encode(), jpg_bytes])
                    self.stats['sent'] += 1
                except:
                    pass
        
        for _ in range(NUM_ENCODER_PROCESSES):
            try:
                self.encode_input_queue.put_nowait(None)
            except:
                pass
        
        for p in self.encoders:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        
        self._log(final=True)
        print("🛑 FastCameraSender stopped")
    
    def _maybe_log(self):
        now = time.time()
        
        # 1초마다 FPS 계산
        if now - self.stats['last_sec'] >= 1.0:
            # 최근 1초간 전송 수
            recent_sends = sum(1 for t in self.sent_times if now - t <= 1.0)
            self.stats['fps_1sec'] = recent_sends
            self.stats['last_sec'] = now
        
        # 3초마다 로그
        if now - self.stats['last_log'] >= 3.0:
            self._log()
            self.stats['last_log'] = now
    
    def _log(self, final=False):
        tag = "FINAL" if final else "STAT"
        s = self.stats
        
        # 평균 인코딩 시간
        avg_encode = 0.0
        if s['encode_samples'] > 0:
            avg_encode = s['total_encode_time'] / s['encode_samples']
            s['total_encode_time'] = 0.0
            s['encode_samples'] = 0
        
        # 최근 3초 FPS
        now = time.time()
        fps_3sec = sum(1 for t in self.sent_times if now - t <= 3.0) / 3.0
        
        print(f"[CAM-{tag}] sub={s['submitted']} | enc={s['encoded']} | "
              f"sent={s['sent']} | drop_enc={s['dropped_encode']} | "
              f"drop_snd={s['dropped_send']} | "
              f"fps_1s={s['fps_1sec']:.1f} | fps_3s={fps_3sec:.1f} | "
              f"avg_enc={avg_encode:.1f}ms | "
              f"q_in={self.encode_input_queue.qsize()} | "
              f"q_out={self.encode_output_queue.qsize()}")

# ===================== ZED 카메라 =====================
class ZedCamera(threading.Thread):
    def __init__(self, serial, name, sender, trigger, send_right=False):
        super().__init__(daemon=True)
        self.sn = int(serial)
        self.name = name
        self.sender = sender
        self.trigger = trigger
        self.send_right = send_right
        
        self.zed = sl.Camera()
        self.runtime = sl.RuntimeParameters()
        self.runtime.enable_depth = False  # Depth 비활성화
        self.runtime.confidence_threshold = 100
        self.runtime.texture_confidence_threshold = 100
        self.runtime.remove_saturated_areas = False
        self.left = sl.Mat()
        self.right = sl.Mat()
        self.ready = False
        self.frame_count = 0
    
    def init(self):
        p = sl.InitParameters()
        p.camera_resolution = sl.RESOLUTION.HD1080
        p.camera_fps = 30  # 60 → 15fps (안정성)
        p.depth_mode = sl.DEPTH_MODE.NONE
        p.camera_image_flip = sl.FLIP_MODE.OFF
        p.camera_disable_self_calib = True  # 자동 캘리브레이션 비활성화
        p.enable_image_enhancement = False  # 이미지 향상 비활성화 (성능)
        p.grab_compute_capping_fps = 0  # CPU 제한 없음
        p.set_from_serial_number(self.sn)
        
        if self.zed.open(p) == sl.ERROR_CODE.SUCCESS:
            self.ready = True
            print(f"✅ ZED {self.sn} ready ({self.name}, 15fps)")
            return True
        else:
            print(f"❌ ZED {self.sn} init failed")
            return False
    
    def run(self):
        if not self.ready:
            return
        
        print(f"🎥 ZED {self.sn} waiting for 2Hz trigger...")
        last_trigger = False
        
        while not stop_flag.is_set():
            current = self.trigger.event.is_set()
            
            # Rising edge
            if current and not last_trigger:
                if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
                    ts = time.time()
                    
                    self.zed.retrieve_image(self.left, sl.VIEW.LEFT)
                    left_np = self.left.get_data()[:, :, :3].copy()
                    self.sender.submit(f"zed_{self.sn}_left", left_np, ts)
                    
                    if self.send_right:
                        self.zed.retrieve_image(self.right, sl.VIEW.RIGHT)
                        right_np = self.right.get_data()[:, :, :3].copy()
                        self.sender.submit(f"zed_{self.sn}_right", right_np, ts)
                    
                    self.frame_count += 1
            
            last_trigger = current
            time.sleep(0.0002)  # 0.2ms (2Hz에 충분)
        
        self.zed.close()
        print(f"🛑 ZED {self.sn} stopped (frames: {self.frame_count})")

# ===================== OAK 카메라 =====================
class OakCamera(threading.Thread):
    def __init__(self, camera_socket, sender, trigger):
        super().__init__(daemon=True)
        self.camera_socket = camera_socket
        self.sender = sender
        self.trigger = trigger
        self.frame_count = 0
    
    def run(self):
        pipeline = dai.Pipeline()
        
        cam = pipeline.createColorCamera()
        cam.setBoardSocket(getattr(dai.CameraBoardSocket, self.camera_socket))
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setFps(60)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.initialControl.setManualFocus(105)
        
        xout = pipeline.createXLinkOut()
        xout.setStreamName("video")
        cam.video.link(xout.input)
        
        try:
            with dai.Device(pipeline) as device:
                q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
                print(f"✅ OAK {self.camera_socket} ready (2Hz mode)")
                
                last_trigger = False
                
                while not stop_flag.is_set():
                    current = self.trigger.event.is_set()
                    
                    if current and not last_trigger:
                        frame = None
                        while q_video.has():
                            frame = q_video.get()
                        
                        if frame is not None:
                            img = frame.getCvFrame()
                            ts = time.time()
                            self.sender.submit(f"oak_{self.camera_socket}", img, ts)
                            self.frame_count += 1
                    
                    last_trigger = current
                    time.sleep(0.0002)
        
        except Exception as e:
            print(f"[OAK] Error: {e}")
        
        print(f"🛑 OAK {self.camera_socket} stopped (frames: {self.frame_count})")

# ===================== 메인 =====================
def main():
    print("\n" + "="*80)
    print("🚀 Multi-Camera Sender - 2Hz Mode (2 frames/sec)")
    print("   - Capture: Every 0.5 seconds")
    print("   - Expected: ~10 frames/sec total (5 cameras)")
    print("="*80 + "\n")
    
    # 전송 시스템
    sender = FastCameraSender(
        ip=SERVER_IP,
        port=SERVER_PORT,
        quality=JPEG_QUALITY,
        resize=RESIZE_BEFORE_ENCODE
    )
    sender.start()
    
    # 2Hz 트리거
    trigger = HighFreqTrigger(
        interval=CAPTURE_INTERVAL,
        pulse_width=PULSE_WIDTH
    )
    trigger.start()
    
    # 카메라
    cameras = []
    
    zed_configs = [
        (41182735, "View1"),
        (49429257, "View2"),
        (44377151, "View3"),
        (49045152, "View4")
    ]
    
    for serial, name in zed_configs:
        zed = ZedCamera(serial, name, sender, trigger, SEND_ZED_RIGHT)
        if zed.init():
            cameras.append(zed)
            zed.start()
    
    oak = OakCamera("CAM_A", sender, trigger)
    cameras.append(oak)
    oak.start()
    
    print(f"\n✅ System started:")
    print(f"   📷 Cameras: {len(cameras)} active")
    print(f"   ⏱  Rate: 2 Hz (0.5s interval)")
    print(f"   📊 Expected throughput: ~10 frames/sec")
    print(f"   💾 Quality: {JPEG_QUALITY} | Resize: {RESIZE_BEFORE_ENCODE}")
    print(f"   🔧 Encoders: {NUM_ENCODER_PROCESSES} processes")
    print("\nPress Ctrl+C to stop...\n")
    
    try:
        while not stop_flag.is_set():
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    
    finally:
        print("\n--- Shutdown ---")
        stop_flag.set()
        encoder_stop_flag.set()
        
        for cam in cameras:
            cam.join(timeout=2.0)
        
        sender.join(timeout=5.0)
        
        print("\n✅ Shutdown complete")

if __name__ == "__main__":
    main()