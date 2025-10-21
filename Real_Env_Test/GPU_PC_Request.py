# A6000_controller.py
import zmq, json, time

JETSON_IP = "10.130.4.81"   # Jetson의 실제 IP
PORT = 6000

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect(f"tcp://{JETSON_IP}:{PORT}")

# 촬영 시작 명령
sock.send_json({"cmd": "start"})
print("📤 Sent start command")
print(sock.recv_json())

time.sleep(10)  # 10초 후 중단 명령
sock.send_json({"cmd": "stop"})
print("📤 Sent stop command")
print(sock.recv_json())
