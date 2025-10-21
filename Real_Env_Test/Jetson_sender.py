import os, time, json, cv2, zmq, numpy as np, threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import depthai as dai
import pyzed.sl as sl

# ===================== 설정 =====================
SERVER_IP = "10.130.4.79"
SERVER_PORT = 5555

# ✅ 품질 75 권장 (시각적으로 양호, 인코딩/전송/저장 부담↓). 필요시 60~85에서 조절.
JPEG_QUALITY = 85

OUTPUT_DIR = "./dataset/ZED_Captures"
SEND_ZED_RIGHT = True
CAPTURE_INTERVAL = 1 # ✅ 1초마다 트리거
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 트리거 스레드 =====================
class CaptureTrigger(threading.Thread):
    """interval마다 모든 카메라에 캡처 신호(event)를 흘려보냄."""
    def __init__(self, interval=1.0, pulse_width=0.03):
        super().__init__(daemon=True)
        self.interval = float(interval)
        self.pulse_width = float(pulse_width)
        self.event = threading.Event()
        self.stop_flag = threading.Event()

    def run(self):
        print(f"⏱ CaptureTrigger started (interval={self.interval:.3f}s)")
        while not self.stop_flag.is_set():
            self.event.set()
            time.sleep(self.pulse_width)   # 신호 유지
            self.event.clear()
            # 남은 시간 대기
            remain = self.interval - self.pulse_width
            if remain > 0:
                time.sleep(remain)
        print("🛑 CaptureTrigger stopped.")

# ===================== 비동기 전송(멀티스레드 인코딩) =====================
class AsyncSender(threading.Thread):
    """
    - 생산자(카메라 스레드)가 submit()으로 프레임을 밀어넣음
    - run()은 큐에서 꺼내 ThreadPoolExecutor에 인코딩/송신 작업 위임
    - 큐가 차면 가장 오래된 항목을 간단히 드롭하여 백프레셔 완화
    """
    def __init__(self, ip, port, quality=75, max_q=400, enc_workers=4, zmq_io_threads=4, snd_hwm=2000):
        super().__init__(daemon=True)
        self.ctx = zmq.Context.instance()
        # ✅ ZMQ I/O threads 증가 (동시 송신 병렬성↑)
        self.ctx.setsockopt(zmq.IO_THREADS, zmq_io_threads)

        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, snd_hwm)
        self.sock.connect(f"tcp://{ip}:{port}")

        self.enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        self.q = Queue(max_q)
        self.max_q = max_q
        self.stop_evt = threading.Event()

        # ✅ 병렬 인코딩 풀
        self.pool = ThreadPoolExecutor(max_workers=enc_workers)

        # 통계
        self.sent = 0
        self.dropped_full = 0
        self.dropped_backpressure = 0
        self.last_log = time.time()

    def submit(self, cam, frame, ts):
        if self.stop_evt.is_set():
            return
        # 큐가 거의 꽉 찼으면 가장 오래된 것을 버려서 backpressure 완화
        if self.q.qsize() > int(self.max_q * 0.9):
            try:
                _ = self.q.get_nowait()
                self.dropped_backpressure += 1
            except Empty:
                pass
        try:
            self.q.put_nowait((cam, frame, ts))
        except:
            self.dropped_full += 1
            # 최후의 방어(가득차면 가장 오래된 것 날리고 새것을 넣어본다)
            try:
                _ = self.q.get_nowait()
                self.q.put_nowait((cam, frame, ts))
            except:
                pass

    def _encode_and_send(self, cam, frame, ts):
        ok, buf = cv2.imencode(".jpg", frame, self.enc)
        if not ok:
            return
        meta = {
            "camera": cam,
            "timestamp": float(ts),      # 유닉스 시각
            "send_time": round(time.time(), 3),
            "size": int(buf.nbytes),
        }
        try:
            self.sock.send_multipart([json.dumps(meta).encode(), buf.tobytes()], flags=zmq.NOBLOCK)
            self.sent += 1
        except zmq.Again:
            self.dropped_full += 1

    def run(self):
        print("📡 AsyncSender started.")
        while not self.stop_evt.is_set():
            try:
                cam, frame, ts = self.q.get(timeout=0.05)
            except Empty:
                # 주기적으로 통계 로그
                self._maybe_log()
                continue

            # 인코딩/송신을 스레드풀로 위임
            self.pool.submit(self._encode_and_send, cam, frame, ts)
            self._maybe_log()

        # 종료 시 큐 비우고 남은 작업 제출
        while True:
            try:
                cam, frame, ts = self.q.get_nowait()
                self.pool.submit(self._encode_and_send, cam, frame, ts)
            except Empty:
                break

        self.pool.shutdown(wait=True)
        self._log(final=True)
        print("🛑 AsyncSender stopped.")

    def _maybe_log(self):
        if time.time() - self.last_log >= 5.0:
            self._log()
            self.last_log = time.time()

    def _log(self, final=False):
        tag = "FINAL" if final else "STAT"
        print(f"[{tag}] sent={self.sent} | q={self.q.qsize()} | drop(full)={self.dropped_full} | drop(bp)={self.dropped_backpressure}")

# ===================== ZED =====================
class ZedCamera(threading.Thread):
    def __init__(self, serial, name, sender, trigger, send_right=False):
        super().__init__(daemon=True)
        self.sn, self.name = int(serial), name
        self.sender = sender
        self.trigger = trigger
        self.send_right = send_right
        self.zed = sl.Camera()
        self.runtime = sl.RuntimeParameters()
        self.left = sl.Mat()
        self.right = sl.Mat()
        self.ready = False

    def init(self):
        p = sl.InitParameters()
        p.camera_resolution = sl.RESOLUTION.HD1080
        p.camera_fps = 60            # ← 사용자 요구: 15fps 고정
        p.depth_mode = sl.DEPTH_MODE.NONE
        p.set_from_serial_number(self.sn)
        if self.zed.open(p) == sl.ERROR_CODE.SUCCESS:
            self.ready = True
            print(f"✅ ZED {self.sn} ready")

    def run(self):
        if not self.ready:
            return
        print(f"🎥 ZED {self.sn} waiting for trigger")

        # "하나의 트리거 → 정확히 1장" 보장: edge 감지
        last_state = False
        while not self.sender.stop_evt.is_set():
            cur = self.trigger.event.is_set()
            # rising edge 감지
            if cur and not last_state:
                if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
                    t = round(time.time(), 3)
                    self.zed.retrieve_image(self.left, sl.VIEW.LEFT)
                    left_np = self.left.get_data()[:, :, :3]
                    self.sender.submit(f"zed_{self.sn}_left", left_np, t)

                    if self.send_right:
                        self.zed.retrieve_image(self.right, sl.VIEW.RIGHT)
                        right_np = self.right.get_data()[:, :, :3]
                        self.sender.submit(f"zed_{self.sn}_right", right_np, t)
            last_state = cur
            time.sleep(0.001)  # CPU 과점유 방지

        print(f"🛑 ZED {self.sn} stopped")

# ===================== OAK =====================
def run_oak(camera_socket="RGB", sender=None, trigger=None):
    """
    트리거의 rising edge마다 q_video에서 가장 최신 프레임을 가져와 한 장만 전송.
    """
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setBoardSocket(getattr(dai.CameraBoardSocket, camera_socket))
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(60)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.initialControl.setManualFocus(110)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("video")
    cam.video.link(xout.input)

    with dai.Device(pipeline) as device:
        q_video = device.getOutputQueue(name="video", maxSize=8, blocking=False)
        print(f"✅ OAK {camera_socket} ready (trigger-based mode)")

        last_state = False
        while not sender.stop_evt.is_set():
            cur = trigger.event.is_set()
            if cur and not last_state:
                # rising edge: 큐에서 가장 최신 프레임 하나만 사용 (구형 프레임 버림)
                frame = q_video.get() if q_video.has() else None
                # 쌓여있으면 마지막 것만 쓰고 나머지는 자연 폐기되도록(다음 루프에서 소비)
                if frame is not None:
                    img = frame.getCvFrame()
                    t = round(time.time(), 3)
                    sender.submit(f"oak_{camera_socket}", img, t)
            last_state = cur
            time.sleep(0.001)

    print(f"🛑 OAK {camera_socket} stopped")

# ===================== 메인 =====================
def main():
    sender = AsyncSender(
        SERVER_IP, SERVER_PORT,
        quality=JPEG_QUALITY,
        max_q=400,
        enc_workers=6,       # CPU 코어에 맞춰 4~6 범위 권장
        zmq_io_threads=4,    # ZMQ I/O threads 증가
        snd_hwm=2000
    )
    sender.start()

    trigger = CaptureTrigger(interval=CAPTURE_INTERVAL, pulse_width=0.03)
    trigger.start()

    # OAK
    oak_thr = threading.Thread(
        target=run_oak,
        kwargs={"camera_socket": "CAM_A", "sender": sender, "trigger": trigger},
        daemon=True
    )
    oak_thr.start()

    # ZED 카메라들
    zeds = [
        ZedCamera(41182735, "view1", sender, trigger, send_right=SEND_ZED_RIGHT),
        ZedCamera(49429257, "view2", sender, trigger, send_right=SEND_ZED_RIGHT),
        ZedCamera(44377151, "view3", sender, trigger, send_right=SEND_ZED_RIGHT),
        ZedCamera(49045152, "view4", sender, trigger, send_right=SEND_ZED_RIGHT),
    ]
    for z in zeds: z.init()
    for z in zeds: z.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Shutting down...")
        sender.stop_evt.set()
        trigger.stop_flag.set()

    for z in zeds:
        z.join(timeout=1.0)
    sender.join()
    print("✅ Exit")

if __name__ == "__main__":
    main()
