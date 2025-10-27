import os
import sys
import time
import csv
import random
import argparse
import logging
import pathlib
import threading
import struct # Keep struct for packing the payload
# import numpy as np # No longer needed here for padding
import zmq # Added for ZeroMQ

import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

# =========================
# Configuration
# =========================
# 저장 폴더
OUTPUT_DIR = "./dataset/Robot_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ZeroMQ Publisher Configuration
ZMQ_PUB_ADDRESS = "*" # Bind to all interfaces on this machine
ZMQ_PUB_PORT = 5556 # Choose a port for ZMQ
SENDER_RATE_HZ = 100 # How often to send ZMQ messages
ZMQ_TOPIC = b"robot_state" # Topic for subscribers to filter

# 데이터 포맷 (C++ receiver와 일치)
# Format: <ddf12f
# - origin_timestamp: double (8 bytes)
# - send_timestamp: double (8 bytes)
# - force: float (4 bytes)
# - joints[6]: float (24 bytes)
# - pose[6]: float (24 bytes)
# Total: 68 bytes
PAYLOAD_FORMAT = '<ddf12f'
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FORMAT)

# ============================================================
# 1️⃣ Global Clock (유지)
# ============================================================
class GlobalClock(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.timestamp = round(time.time(), 3)
        self.running = True
        self.lock = threading.Lock()

    def now(self):
        with self.lock:
            return self.timestamp

    def run(self):
        while self.running:
            with self.lock:
                self.timestamp = round(time.time(), 3)
            time.sleep(0.005) # Update clock at ~200Hz

    def stop(self):
        self.running = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", default=None, help="출력 폴더 접미사 (예: 20th)")
    p.add_argument("--robot", choices=["on", "off"], default="on",
                   help="로봇 제어 활성화 여부 (기본값: on)")
    return p.parse_args()


# ============================================================
# 4️⃣ 실시간 로봇 데이터 샘플러 (유지)
# ============================================================
class RtSampler(threading.Thread):
    def __init__(self, robot, out_csv, clock, rate_hz=100):
        super().__init__(daemon=True)
        self.robot = robot
        self.out_csv = out_csv
        self.dt = 1.0 / float(rate_hz)
        self.clock = clock
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.latest_q = None
        self.latest_p = None

    def stop(self):
        self.stop_evt.set()

    def get_latest_data(self):
        with self.lock:
            return self.latest_q, self.latest_p

    def run(self):
        print(f"✅ Starting robot data sampling to {self.out_csv} at {1/self.dt:.1f} Hz...")
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "joint_angle_1", "joint_angle_2", "joint_angle_3",
                "joint_angle_4", "joint_angle_5", "joint_angle_6",
                "EE_x", "EE_y", "EE_z", "EE_a", "EE_b", "EE_r"
            ])
            next_t = time.time()
            while not self.stop_evt.is_set():
                q, p = None, None
                ts_now = self.clock.now()
                for name in ("GetJoints", "GetJointPos", "GetJointAngles"):
                    fn = getattr(self.robot, name, None)
                    if callable(fn):
                        try: q = list(fn()); break
                        except Exception: pass
                for name in ("GetPose", "GetPoseXYZABC", "GetCartesianPose"):
                    fn = getattr(self.robot, name, None)
                    if callable(fn):
                        try: p = list(fn()); break
                        except Exception: pass

                if q is not None and len(q) >= 6 and p is not None and len(p) >= 6:
                    w.writerow([f"{ts_now:.6f}"] + q[:6] + p[:6])
                    with self.lock:
                        self.latest_q = q[:6]
                        self.latest_p = p[:6]

                next_t += self.dt
                sleep_dt = next_t - time.time()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                else:
                    # Limit the frequency of the warning message
                    if random.random() < 0.01: # Print roughly once every 10 seconds if falling behind
                         print(f"[RtSampler WARN] Loop falling behind by {-sleep_dt*1000:.1f} ms")
        print(f"✅ Robot data sampling stopped.")

class ZmqPublisher(threading.Thread):
    def __init__(self, sampler, clock, address, port, stop_event, rate_hz=100):
        super().__init__(daemon=True)
        self.sampler = sampler
        self.clock = clock
        self.address = address
        self.port = port
        self.stop_event = stop_event
        self.dt = 1.0 / float(rate_hz)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        bind_addr = f"tcp://{self.address}:{self.port}"
        self.socket.set_hwm(10)
        self.socket.setsockopt(zmq.LINGER, 500)
        self.socket.bind(bind_addr)
        print(f"✅ ZMQ Publisher bound to {bind_addr} at {rate_hz} Hz.")
        print(f"   Topic: '{ZMQ_TOPIC.decode()}', Payload Size: {PAYLOAD_SIZE} bytes")

    def run(self):
        next_send_time = time.time()
        
        # ▼▼▼ [수정] Create poller and dummy_sock *outside* the loop ▼▼▼
        poller = zmq.Poller()
        dummy_sock = self.context.socket(zmq.PULL)
        # Use unique inproc address per thread instance if needed, though usually not required
        # dummy_addr = f"inproc://dummy_poll_{id(self)}" 
        dummy_addr = "inproc://dummy_poll" # Usually fine
        try:
             dummy_sock.bind(dummy_addr)
             poller.register(dummy_sock, zmq.POLLIN)
        except zmq.ZMQError as e:
            print(f"[ZmqPublisher ERR] Failed to bind dummy socket: {e}")
            # Optionally handle this - maybe fall back to time.sleep?
            dummy_sock = None # Indicate failure
        # ▲▲▲ [수정] ▲▲▲

        try: # Added try block for better cleanup
            while not self.stop_event.is_set():
                q, p = self.sampler.get_latest_data()

                if q is not None and p is not None:
                    ts = self.clock.now()
                    force = 0.0
                    send_ts = time.time()

                    try:
                        payload_bytes = struct.pack(PAYLOAD_FORMAT, ts, send_ts, force, *q, *p)

                        if len(payload_bytes) != PAYLOAD_SIZE:
                            print(f"[ZmqPublisher ERR] Payload size mismatch!")
                            continue

                        self.socket.send_multipart([ZMQ_TOPIC, payload_bytes], zmq.DONTWAIT)

                    except zmq.Again:
                        pass
                    except zmq.ZMQError as e:
                        print(f"[ZmqPublisher ERR] Failed to send ZMQ message: {e}")
                        if e.errno == zmq.ETERM: break
                    except Exception as e:
                        print(f"[ZmqPublisher ERR] Unexpected error during send: {e}")

                # Sleep to maintain rate
                next_send_time += self.dt
                sleep_duration = next_send_time - time.time()
                
                if sleep_duration > 0 and dummy_sock: # Check if dummy_sock was created
                    try:
                        # ▼▼▼ [수정] No need to register/unregister inside loop ▼▼▼
                        # poller.register(dummy_sock, zmq.POLLIN) # Moved outside
                        events = poller.poll(int(sleep_duration * 1000))
                        # poller.unregister(dummy_sock) # Moved outside
                        # ▲▲▲ [수정] ▲▲▲
                        if self.stop_event.is_set(): break
                    except zmq.ZMQError as e:
                        # Handle potential errors during poll, e.g., context termination
                        if e.errno == zmq.ETERM: break
                        print(f"[ZmqPublisher WARN] Poller error: {e}")
                        # Fallback sleep if poller fails unexpectedly
                        time.sleep(max(0, sleep_duration))
                    except Exception as e:
                         print(f"[ZmqPublisher WARN] Unexpected error during poll/sleep: {e}")
                         time.sleep(max(0, sleep_duration))

                elif sleep_duration > 0: # Fallback if dummy_sock failed
                    time.sleep(sleep_duration)


        finally: # Ensure cleanup happens
             print("🧹 Cleaning up ZMQ Publisher...")
             # ▼▼▼ [수정] Clean up dummy_sock ▼▼▼
             if dummy_sock:
                 # Check if registered before trying to unregister
                 try:
                      poller.unregister(dummy_sock)
                 except KeyError: # Already unregistered or never registered
                      pass
                 except Exception as e:
                      print(f"[ZmqPublisher WARN] Error unregistering dummy_sock: {e}")
                 dummy_sock.close()
             # ▲▲▲ [수정] ▲▲▲
             self.socket.close()
             # Check if context termination is needed/safe
             if not self.context.closed:
                 self.context.term()
             print("✅ ZMQ Publisher stopped.")


# ============================================================
# 5️⃣ 로봇 매니저 (유지)
# ============================================================
class RobotManager:
    def __init__(self, address="192.168.0.100"):
        self.address = address
        self.robot = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        # Configure logging
        log_file = f'{pathlib.Path(__file__).stem}.log'
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(log_file),
                                      logging.StreamHandler(sys.stdout)]) # Log to file and console
        self.logger.info(f"Log file: {log_file}")
        # tools.SetDefaultLogger(logging.INFO, log_file) # Use basicConfig instead

        self.robot = initializer.RobotWithTools()
        self.robot.__enter__()
        self.logger.info(f"Connecting to robot at {self.address}...")
        try:
            self.robot.Connect(address=self.address, disconnect_on_exception=False)
            self.logger.info("Robot connected.")
        except mdr.MecademicException as e:
            self.logger.error(f"Failed to connect to robot: {e}")
            raise # Re-raise exception to stop the program if connection fails
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.robot and self.robot.IsConnected():
            self.logger.info("Disconnecting robot...")
            try:
                # Check error status before deactivating
                status = self.robot.GetStatusRobot()
                if status.error_status:
                    self.logger.warning('Robot is in error state, attempting to reset...')
                    self.robot.ResetError()
                    time.sleep(0.5) # Allow time for reset
                    self.robot.ResumeMotion() # Might be needed after reset
                    time.sleep(0.5)
            except Exception as e:
                self.logger.warning(f'Error check/clear failed during exit: {e}')

            try:
                # Attempt to deactivate robot
                self.logger.info("Deactivating robot...")
                self.robot.DeactivateRobot()
            except Exception as e:
                self.logger.warning(f'Deactivate failed during exit: {e}')
        if self.robot:
            self.robot.__exit__(exc_type, exc_value, traceback)
        self.logger.info("Robot disconnected.")
        logging.shutdown() # Ensure logs are flushed

    def setup(self):
        self.logger.info('Activating and homing robot...')
        self.robot.SetJointVel(3)
        initializer.reset_sim_mode(self.robot)
        initializer.reset_motion_queue(self.robot, activate_home=True)
        initializer.reset_vacuum_module(self.robot) # Remove if no vacuum module
        self.robot.WaitHomed()
        self.robot.SetCartLinVel(100)
        self.robot.SetJointVel(1)
        self.robot.SetBlending(50)
        self.robot.WaitIdle(30)
        self.logger.info('Robot setup complete.')

    def move_angle_points(self, points):
        if not tools.robot_model_is_meca500(self.robot.GetRobotInfo().robot_model):
            raise mdr.MecademicException(f'Unsupported robot model: {self.robot.GetRobotInfo().robot_model}')

        for idx, angles in enumerate(points):
            self.logger.info(f'Moving to joint angles {idx+1}: {angles}')
            self.robot.MoveJoints(*angles)
            self.robot.WaitIdle(180) # Wait up to 60 seconds for move to complete

    def move_EE_points(self, points):
        if not tools.robot_model_is_meca500(self.robot.GetRobotInfo().robot_model):
            raise mdr.MecademicException(f'Unsupported robot model: {self.robot.GetRobotInfo().robot_model}')

        self.robot.SetConf(1, 1, 1) # Set configuration for MovePose (Shoulder, Elbow, Wrist)
        for idx, pose in enumerate(points):
            self.logger.info(f'Moving to EE pose {idx+1}: {pose}')
            self.robot.MovePose(*pose)
            self.robot.WaitIdle(180)


# ============================================================
# 6️⃣ 메인 함수 (ZMQ Publisher 사용)
# ============================================================
def main():
    args = parse_args()

    global OUTPUT_DIR
    if args.run_tag:
        OUTPUT_DIR = f"./dataset/Robot_Data_{args.run_tag}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Use logging instead of print for consistency
    logging.info(f"Output directory set to: {OUTPUT_DIR}")

    stop_event = threading.Event()
    clock = GlobalClock()
    clock.start()

    sampler = None
    sender = None

    try:
        if args.robot == "on":
            logging.info("Robot mode is ON.")
            with RobotManager() as manager: # This handles connection and disconnection
                manager.setup()
                sampler_csv = os.path.join(OUTPUT_DIR, f"robot_rt_{clock.now():.3f}.csv")
                sampler = RtSampler(manager.robot, sampler_csv, clock, rate_hz=100)
                sampler.start()

                # --- Start ZMQ Publisher ---
                sender = ZmqPublisher(sampler, clock, ZMQ_PUB_ADDRESS, ZMQ_PUB_PORT, stop_event, rate_hz=SENDER_RATE_HZ)
                sender.start()
                # ---

                logging.info("Starting robot movement...")
                manager.move_angle_points([
                    # # 눈 트로카 진입
                    # (51.093, -0.2565, 27.4365, -1.001, 24.416, -34.72),
                    # (51.045512, 6.187758, 28.528616, -1.326412, 16.839833, -34.399135),
                    # White silcone 정 중앙 진입
                    (-0.338223, 1.107869, -2.314018, -0.304322, 70.844049, -2.447558),
                    (1.439584, 7.482698, 5.040527, -0.815003, 59.736926, -0.392272)
                ])
                logging.info("First movement sequence finished.")

                # --- Stop Sampler and Sender ---
                logging.info("Signaling sampler and sender to stop...")
                stop_event.set() # Signal threads to stop after movements
                # Join is handled in the finally block
                # ---

                # Move back and to noisy home AFTER signaling stop
                # This ensures the threads capture the end of the main movement
                logging.info("Moving back to initial position...")
                manager.robot.SetJointVel(2)
                manager.move_angle_points([
                    # 눈 트로카 후퇴
                    # (51.093, -0.2565, 27.4365, -1.001, 24.416, -34.72),
                    # White silcone 정 중앙 후퇴
                    (-0.338223, 1.107869, -2.314018, -0.304322, 70.844049, -2.447558)
                ])
                noise = random.uniform(-15.0, 15.0)
                logging.info(f"Moving to noisy home position with noise: {noise:.2f}...")
                home_pose_noisy = (190.0+noise, 0.0+noise, 308.0+noise, 0.0, 90.0, 0.0)
                manager.move_EE_points([home_pose_noisy])
                logging.info("Robot movements complete.")

        else:
            logging.info("Robot mode is OFF. Script will idle. Press Ctrl+C to exit.")
            while not stop_event.is_set():
                try: time.sleep(1)
                except KeyboardInterrupt: logging.info("\nCtrl+C detected."); stop_event.set()

    except KeyboardInterrupt:
        logging.info("\nCtrl+C detected during robot operation. Stopping threads...")
        stop_event.set()
    except mdr.MecademicException as e:
        logging.error(f"\n--- Mecademic Robot Error: {e} ---")
        stop_event.set()
    except Exception as e:
        logging.error(f"\n--- An unexpected error occurred: {e} ---")
        logging.exception("Error details:") # Log traceback
        stop_event.set()
    finally:
        # --- Ensure threads are stopped and joined ---
        if not stop_event.is_set():
             logging.info("Setting stop event in finally block...")
             stop_event.set()

        # Stop the clock thread first
        clock.stop()
        logging.info("Clock stopped.")

        # Join threads (wait for them to finish)
        # It's generally safer to join the sender first as it depends on the sampler
        if sender and sender.is_alive():
            logging.info("Waiting for sender thread to finish...")
            sender.join(timeout=5.0)
            if sender.is_alive(): logging.warning("Sender thread did not exit cleanly.")
            else: logging.info("Sender thread finished.")

        if sampler and sampler.is_alive():
            logging.info("Waiting for sampler thread to finish (CSV writing)...")
            sampler.join(timeout=5.0)
            if sampler.is_alive(): logging.warning("Sampler thread did not exit cleanly.")
            else: logging.info("Sampler thread finished.")
        # ---

        logging.info("Data collection script finished.")


if __name__ == "__main__":
    main()