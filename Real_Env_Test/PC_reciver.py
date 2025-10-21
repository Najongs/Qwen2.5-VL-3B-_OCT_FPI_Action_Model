import os, time, json, cv2, zmq, numpy as np
from collections import defaultdict

SAVE_DIR = "./recv_all"
os.makedirs(SAVE_DIR, exist_ok=True)

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PULL)
sock.setsockopt(zmq.RCVHWM, 5000)
sock.bind("tcp://0.0.0.0:5555")
print("✅ Listening on tcp://0.0.0.0:5555")

cnt = defaultdict(int); t0 = time.time()
while True:
    parts = sock.recv_multipart()
    if len(parts) < 2:
        # 방어: 혹시라도 1-part가 오면 스킵
        continue
    meta = json.loads(parts[0].decode("utf-8"))
    jpg  = parts[1]
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

    cam = meta.get("camera","unknown")
    ts  = meta.get("timestamp", time.time())
    cnt[cam] += 1

    # 저장(옵션)
    fn = os.path.join(SAVE_DIR, f"{cam}_{ts:.3f}.jpg")
    cv2.imwrite(fn, img)

    # 주기 출력
    if time.time() - t0 > 2:
        line = " | ".join([f"{k}:{cnt[k]/(time.time()-t0):.1f}fps" for k in sorted(cnt)])
        print(line)
        cnt = defaultdict(int); t0 = time.time()
