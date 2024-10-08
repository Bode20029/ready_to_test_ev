import cv2
import time
import logging
import pygame
from pymongo import MongoClient
from gridfs import GridFS
from line_notify import LineNotifier
from dotenv import load_dotenv
import os
from datetime import datetime
import threading
import queue
import Jetson.GPIO as GPIO
import serial
import json
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu
import numpy as np
from ultralytics import YOLO

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DISTANCE_THRESHOLD = 150  # cm
CURRENT_THRESHOLD = 0.1  # A
DETECTION_STABILITY_TIME = 5  # seconds
CHARGING_CHECK_TIME = 600  # 10 minutes
ALARM_INTERVALS = [180, 300, 600]  # 3 minutes, 5 minutes, 10 minutes
LINE_NOTIFY_TOKEN = "J0oQ74OftbCNdiPCCfV4gs75aqtz4aAL8NiGfHERvZ4"
# MONGO_URI = os.getenv("MONGO_URI")
MONGO_URI = "mongodb+srv://Extremenop:Nop24681036@cardb.ynz57.mongodb.net/?retryWrites=true&w=majority&appName=Cardb"
DB_NAME = "cardb"
COLLECTION_NAME = "nonev"

# HC-SR04P Constants
TRIG_PIN = 12
ECHO_PIN = 16
DISTANCE_MEASUREMENT_INTERVAL = 0.1

# PZEM Constants
PZEM_PORT = '/dev/ttyUSB0'
PZEM_MEASUREMENT_INTERVAL = 1  # 1 second

class EVMonitoringSystem:
    def __init__(self):
        self.camera = None
        self.yolo_model = YOLO('yolov8n.pt')  # Initialize YOLO model
        self.line_notifier = None
        self.client = None
        self.db = None
        self.collection = None
        self.fs = None
        self.distance_sensor_data = queue.Queue()
        self.pzem_sensor_data = queue.Queue()
        self.stop_event = threading.Event()
        self.process_active = False
        self.vehicle_present = False
        self.is_ev = False
        self.current_process = None

        pygame.mixer.init()

    def initialize_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            return False
        logger.info("Camera initialized successfully")
        return True

    def initialize_line(self):
        try:
            self.line_notifier = LineNotifier()
            self.line_notifier.token = LINE_NOTIFY_TOKEN
            logger.info("LINE Notifier initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LINE Notifier: {e}")
            return False

    def initialize_mongodb(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            self.fs = GridFS(self.db)
            self.client.server_info()
            logger.info("MongoDB initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            return False

    def initialize_services(self):
        return (self.initialize_camera() and 
                self.initialize_line() and 
                self.initialize_mongodb())

    def detect_vehicle(self, frame):
        results = self.yolo_model(frame)
        for r in results:
            for c in r.boxes.cls:
                class_name = self.yolo_model.names[int(c)]
                if class_name in ['car', 'truck']:  # Adjust based on your YOLO model's classes
                    return True, class_name == 'car'  # True if vehicle detected, True if it's a car (potential EV)
        return False, False

    def play_sound(self, sound_file):
        try:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except pygame.error as e:
            logger.error(f"Error playing sound {sound_file}: {e}")

    def save_image_to_gridfs(self, frame, timestamp):
        _, img_encoded = cv2.imencode('.jpg', frame)
        return self.fs.put(img_encoded.tobytes(), filename=f"event_{timestamp}.jpg", metadata={"timestamp": timestamp})

    def save_metadata(self, file_id, timestamp, event):
        metadata = {
            "file_id": file_id,
            "timestamp": timestamp,
            "event": event
        }
        return self.collection.insert_one(metadata)

    def handle_detection(self, frame, event):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"{timestamp} {event}"

        if event == "Non-EV car parking":
            self.play_sound("alert.mp3")
            self.play_sound("Warning.mp3")
        elif event == "EV car not charging":
            self.play_sound("not_charging.mp3")

        local_path = f"detected_event_{timestamp.replace(':', '-')}.jpg"
        cv2.imwrite(local_path, frame)

        try:
            file_id = self.save_image_to_gridfs(frame, timestamp)
            metadata_id = self.save_metadata(file_id, timestamp, event)
            logger.info(f"Image saved to GridFS with ID: {file_id}")
            logger.info(f"Metadata saved with ID: {metadata_id.inserted_id}")
        except Exception as e:
            logger.error(f"Failed to save image or metadata: {e}")

        try:
            self.line_notifier.send_image(message, local_path)
            logger.info("Line notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Line notification: {e}")

    def get_distance(self):
        GPIO.output(TRIG_PIN, GPIO.LOW)
        time.sleep(0.1)

        GPIO.output(TRIG_PIN, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, GPIO.LOW)

        pulse_start = pulse_end = time.time()
        timeout = pulse_start + 1.0

        while GPIO.input(ECHO_PIN) == GPIO.LOW:
            pulse_start = time.time()
            if pulse_start > timeout:
                return None

        while GPIO.input(ECHO_PIN) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end > timeout:
                return None

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        return round(distance, 2)

    def distance_sensor_thread(self):
        while not self.stop_event.is_set():
            dist = self.get_distance()
            if dist is not None:
                self.distance_sensor_data.put(dist)
            time.sleep(DISTANCE_MEASUREMENT_INTERVAL)

    def connect_to_pzem(self):
        try:
            ser = serial.Serial(
                port=PZEM_PORT,
                baudrate=9600,
                bytesize=8,
                parity='N',
                stopbits=1,
                xonxoff=0
            )
            master = modbus_rtu.RtuMaster(ser)
            master.set_timeout(2.0)
            master.set_verbose(True)
            logger.info("Successfully connected to PZEM sensor")
            return master
        except Exception as e:
            logger.error(f"Failed to connect to PZEM sensor: {e}")
            return None

    def read_pzem_data(self, master):
        data = master.execute(1, cst.READ_INPUT_REGISTERS, 0, 10)
        return {
            "voltage": data[0] / 10.0,
            "current_A": (data[1] + (data[2] << 16)) / 1000.0,
            "power_W": (data[3] + (data[4] << 16)) / 10.0,
            "energy_Wh": data[5] + (data[6] << 16),
            "frequency_Hz": data[7] / 10.0,
            "power_factor": data[8] / 100.0,
            "alarm": data[9]
        }

    def pzem_sensor_thread(self):
        master = self.connect_to_pzem()
        while not self.stop_event.is_set():
            if master is None:
                logger.warning("PZEM sensor not connected. Attempting to reconnect...")
                master = self.connect_to_pzem()
                time.sleep(5)
                continue

            try:
                data = self.read_pzem_data(master)
                self.pzem_sensor_data.put(data)
            except Exception as e:
                logger.error(f"Error reading PZEM data: {e}")
                master.close()
                master = self.connect_to_pzem()
            
            time.sleep(PZEM_MEASUREMENT_INTERVAL)

    def monitor_charging(self):
        start_time = time.time()
        last_current = None
        while time.time() - start_time < CHARGING_CHECK_TIME:
            try:
                pzem_data = self.pzem_sensor_data.get(timeout=1)
                current = pzem_data['current_A']
                if last_current is not None and abs(current - last_current) >= CURRENT_THRESHOLD:
                    logger.info("Charging detected")
                    return True
                last_current = current
            except queue.Empty:
                pass
            if not self.vehicle_present:
                logger.info("Vehicle no longer present")
                return False
        logger.warning("No charging detected in 10 minutes")
        return False

    def process_ev(self):
        if self.monitor_charging():
            logger.info("EV is charging normally")
        else:
            logger.warning("EV not charging")
            ret, frame = self.camera.read()
            if ret:
                self.handle_detection(frame, "EV car not charging")
            self.alarm_sequence()

    def process_non_ev(self):
        logger.warning("Non-EV car detected")
        ret, frame = self.camera.read()
        if ret:
            self.handle_detection(frame, "Non-EV car parking")
        self.alarm_sequence()

    def alarm_sequence(self):
        for interval in ALARM_INTERVALS:
            time.sleep(interval)
            if not self.vehicle_present:
                logger.info("Vehicle no longer present")
                return
            logger.warning("Vehicle still present, sounding alarm again")
            self.play_sound("alert.mp3")

    def run(self):
        if not self.initialize_services():
            logger.error("Failed to initialize services. Exiting.")
            return

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)

        distance_thread = threading.Thread(target=self.distance_sensor_thread)
        pzem_thread = threading.Thread(target=self.pzem_sensor_thread)
        distance_thread.start()
        pzem_thread.start()

        logger.info("Starting main detection loop")
        try:
            while True:
                if not self.camera.isOpened():
                    logger.warning("Camera is not open. Attempting to reopen...")
                    if not self.initialize_camera():
                        logger.error("Failed to reopen camera. Retrying in 5 seconds.")
                        time.sleep(5)
                        continue

                # Check distance sensor
                try:
                    distances = []
                    for _ in range(5):
                        distance = self.distance_sensor_data.get(timeout=1)
                        distances.append(distance)
                    avg_distance = np.mean(distances)
                    self.vehicle_present = avg_distance < DISTANCE_THRESHOLD
                except queue.Empty:
                    logger.warning("No distance data available")

                if self.vehicle_present and not self.process_active:
                    self.process_active = True
                    vehicle_detection_start = time.time()
                    vehicle_detected = False
                    is_ev = False

                    while time.time() - vehicle_detection_start < DETECTION_STABILITY_TIME:
                        ret, frame = self.camera.read()
                        if not ret:
                            continue
                        vehicle_present, is_car = self.detect_vehicle(frame)
                        if not vehicle_present:
                            break
                        vehicle_detected = True
                        is_ev = is_car

                    if vehicle_detected:
                        if is_ev:
                            self.current_process = threading.Thread(target=self.process_ev)
                        else:
                            self.current_process = threading.Thread(target=self.process_non_ev)
                        self.current_process.start()

                elif not self.vehicle_present and self.process_active:
                    logger.info("Vehicle no longer present")
                    self.process_active = False
                    if self.current_process:
                        self.current_process.join()
                    time.sleep(10)  # Wait 10 seconds before starting next detection cycle

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Program stopped by user")
        finally:
            self.stop_event.set()
            distance_thread.join()
            pzem_thread.join()
            self.cleanup()

    def cleanup(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        if self.client:
            self.client.close()
        GPIO.cleanup()
        logger.info("Cleanup completed")

def main():
    system = EVMonitoringSystem()
    system.run()

if __name__ == "__main__":
    main()