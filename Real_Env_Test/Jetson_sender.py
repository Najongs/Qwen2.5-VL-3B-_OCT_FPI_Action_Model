import os, time, json, cv2, zmq, numpy as np, threading
from queue import Queue
import depthai as dai
import pyzed.sl as sl

# ===================== 설정 =====================
SERVER_IP = "10.130.4.79"
SERVER_PORT = 5555
JPEG_QUALITY = 85
OUTPUT_DIR = "./dataset/ZED_Captures"
SEND_ZED_RIGHT = True   # ← 모든 이미지: True면 RIGHT도 전송
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 유틸 =====================
class GlobalClock(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.t=round(time.time(),3); self.runflag=True; self.lock=threading.Lock()
    def now(self):  
        with self.lock: return self.t
    def run(self):
        while self.runflag:
            with self.lock: self.t=round(time.time(),3)
            time.sleep(0.01)
    def stop(self): self.runflag=False

class AsyncImageWriter(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.q=Queue(5000); self.stop_evt=threading.Event()
    def submit(self, path, img): 
        if not self.stop_evt.is_set(): self.q.put((path,img))
    def stop(self): self.stop_evt.set(); self.q.put((None,None))
    def run(self):
        while True:
            p,img=self.q.get()
            if p is None: break
            try: cv2.imwrite(p,img)
            except Exception as e: print(f"[Writer] {p} save err: {e}")

# pyzmq 소켓은 스레드별로 만들어 사용해야 안전
class ThreadSender:
    def __init__(self, ip, port, quality=85):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 1000)
        self.sock.connect(f"tcp://{ip}:{port}")
        self.enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    def send_frame(self, cam, frame, ts):
        ok, buf = cv2.imencode(".jpg", frame, self.enc)
        if not ok: return
        meta = {"camera": cam, "timestamp": ts, "size": int(buf.nbytes)}
        # 항상 2-프레임 멀티파트로 보냄
        self.sock.send_multipart([json.dumps(meta).encode(), buf.tobytes()])

# ===================== ZED =====================
class ZedCamera(threading.Thread):
    def __init__(self, serial, name, clock, writer, send_right=False):
        super().__init__(daemon=True)
        self.sn, self.name = serial, name
        self.clock, self.writer = clock, writer
        self.send_right = send_right
        self.out_dir = os.path.join(OUTPUT_DIR, name); os.makedirs(self.out_dir, exist_ok=True)
        self.zed = sl.Camera(); self.runtime = sl.RuntimeParameters()
        self.left = sl.Mat(); self.right = sl.Mat()
        self.ready=False
        self.sender = ThreadSender(SERVER_IP, SERVER_PORT, JPEG_QUALITY)

    def init(self):
        p = sl.InitParameters()
        p.camera_resolution = sl.RESOLUTION.HD1080
        p.camera_fps = 60
        p.depth_mode = sl.DEPTH_MODE.NONE
        p.set_from_serial_number(self.sn)
        if self.zed.open(p) == sl.ERROR_CODE.SUCCESS:
            self.ready=True
            print(f"✅ ZED {self.sn} ready")

    def run(self):
        if not self.ready: return
        print(f"🎥 ZED {self.sn} start")
        while True:
            if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS: continue
            t = self.clock.now()
            self.zed.retrieve_image(self.left,  sl.VIEW.LEFT)
            left_np  = self.left.get_data()[:, :, :3]
            left_path = os.path.join(self.out_dir, f"zed_{self.sn}_left_{t:.3f}.jpg")
            self.writer.submit(left_path, left_np)
            self.sender.send_frame(f"zed_{self.sn}_left", left_np, t)

            if self.send_right:
                self.zed.retrieve_image(self.right, sl.VIEW.RIGHT)
                right_np = self.right.get_data()[:, :, :3]
                right_path = os.path.join(self.out_dir, f"zed_{self.sn}_right_{t:.3f}.jpg")
                self.writer.submit(right_path, right_np)
                self.sender.send_frame(f"zed_{self.sn}_right", right_np, t)

# ===================== OAK (DepthAI v3 API) =====================
def run_oak(camera_socket="RGB", name="view5_oak"):
    import depthai as dai
    import cv2, time, os, json
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    pipeline = dai.Pipeline()

    # ✅ v2.x 방식: ColorCamera 사용
    cam = pipeline.createColorCamera()
    cam.setBoardSocket(getattr(dai.CameraBoardSocket, camera_socket))
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(60)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("video")
    cam.video.link(xout.input)

    # ✅ 파이프라인 시작
    with dai.Device(pipeline) as device:
        q_video = device.getOutputQueue(name="video", maxSize=8, blocking=False)
        sender = ThreadSender(SERVER_IP, SERVER_PORT, JPEG_QUALITY)
        print(f"✅ OAK {camera_socket} started (DepthAI v2.x API)")

        while True:
            frame = q_video.tryGet()
            if frame is None:
                time.sleep(0.001)
                continue
            frame = frame.getCvFrame()
            t = round(time.time(), 3)
            path = os.path.join(out_dir, f"oak_{camera_socket}_{t:.3f}.jpg")
            cv2.imwrite(path, frame)
            sender.send_frame(f"oak_{camera_socket}", frame, t)



# ===================== 메인 =====================
def main():
    clock = GlobalClock(); clock.start()
    writer = AsyncImageWriter(); writer.start()
    
    # OAK (기본 CAM_A)
    oak_thr = threading.Thread(target=run_oak, kwargs={"camera_socket":"CAM_A","name":"view5_oak"}, daemon=True)
    oak_thr.start()
    
    # ZED 4대
    zeds = [
        ZedCamera(41182735, "view1", clock, writer, send_right=SEND_ZED_RIGHT),
        ZedCamera(49429257, "view2", clock, writer, send_right=SEND_ZED_RIGHT),
        ZedCamera(44377151, "view3", clock, writer, send_right=SEND_ZED_RIGHT),
        ZedCamera(49045152, "view4", clock, writer, send_right=SEND_ZED_RIGHT),
    ]
    
    for z in zeds: z.init()
    for z in zeds: z.start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass

    writer.stop(); writer.join(); clock.stop()
    print("✅ Exit")

if __name__ == "__main__":
    main()
