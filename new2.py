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

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DETECTION_STABILITY_TIME = 5  # seconds
SLEEP_DURATION = 10  # 10 seconds between test cycles
LINE_NOTIFY_TOKEN = "J0oQ74OftbCNdiPCCfV4gs75aqtz4aAL8NiGfHERvZ4"
MONGO_URI = "mongodb+srv://Extremenop:Nop24681036@cardb.ynz57.mongodb.net/?retryWrites=true&w=majority&appName=Cardb"
DB_NAME = "cardb"
COLLECTION_NAME = "nonev"

# HC-SR04P Constants
TRIG_PIN = 12
ECHO_PIN = 16
MIN_DISTANCE = 2
MAX_DISTANCE = 400
TIMEOUT = 1.0
DISTANCE_MEASUREMENT_INTERVAL = 0.1  # 0.1 seconds when no vehicle, will be adjusted when vehicle detected

# PZEM Constants
PZEM_PORT = '/dev/ttyUSB0'
PZEM_MEASUREMENT_INTERVAL = 2  # 2 seconds when no vehicle, will be adjusted when vehicle detected

class IntegratedSystem:
    def __init__(self):
        self.camera = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.line_notifier = None
        self.client = None
        self.db = None
        self.collection = None
        self.fs = None
        self.detection_list = []
        self.distance_sensor_data = queue.Queue()
        self.pzem_sensor_data = queue.Queue()
        self.vehicle_present = False
        self.stop_event = threading.Event()

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

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces) > 0

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
        return self.fs.put(img_encoded.tobytes(), filename=f"face_{timestamp}.jpg", metadata={"timestamp": timestamp})

    def save_metadata(self, file_id, timestamp, event):
        metadata = {
            "file_id": file_id,
            "timestamp": timestamp,
            "event": event
        }
        return self.collection.insert_one(metadata)

    def handle_detection(self, frame):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event = "face_detected"
        message = f"{timestamp} Face detected"

        logger.info("Playing alert sound")
        self.play_sound("alert.mp3")

        local_path = f"detected_face_{timestamp.replace(':', '-')}.jpg"
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
        timeout = pulse_start + TIMEOUT

        while GPIO.input(ECHO_PIN) == GPIO.LOW:
            pulse_start = time.time()
            if pulse_start > timeout:
                logger.warning("Echo pulse start timeout")
                return None

        while GPIO.input(ECHO_PIN) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end > timeout:
                logger.warning("Echo pulse end timeout")
                return None

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150

        if distance < MIN_DISTANCE or distance > MAX_DISTANCE:
            logger.warning(f"Distance out of range: {distance} cm")
            return None

        return round(distance, 2)

    def distance_sensor_thread(self):
        last_valid_distance = None
        while not self.stop_event.is_set():
            for _ in range(3):
                dist = self.get_distance()
                if dist is not None:
                    if last_valid_distance is None or abs(dist - last_valid_distance) < 50:
                        last_valid_distance = dist
                        self.distance_sensor_data.put(dist)
                        break
            else:
                logger.warning("Failed to get a valid distance after 3 attempts")

            time.sleep(DISTANCE_MEASUREMENT_INTERVAL if not self.vehicle_present else DISTANCE_MEASUREMENT_INTERVAL / 2)

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
            
            time.sleep(PZEM_MEASUREMENT_INTERVAL if not self.vehicle_present else PZEM_MEASUREMENT_INTERVAL / 4)

    def run(self):
        if not self.initialize_services():
            logger.error("Failed to initialize services. Exiting.")
            return

        # Initialize GPIO
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

                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame from camera")
                    time.sleep(1)
                    continue

                face_detected = self.detect_face(frame)
                logger.info(f"Face detected: {face_detected}")

                if face_detected:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                cv2.imshow('Video', frame)

                try:
                    distance = self.distance_sensor_data.get_nowait()
                    self.vehicle_present = distance < 100
                    logger.info(f"Distance: {distance} cm")
                except queue.Empty:
                    pass

                try:
                    pzem_data = self.pzem_sensor_data.get_nowait()
                    logger.info(f"PZEM Data: {json.dumps(pzem_data, indent=2)}")
                except queue.Empty:
                    pass

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                self.detection_list.append(face_detected)
                if len(self.detection_list) > DETECTION_STABILITY_TIME:
                    self.detection_list.pop(0)

                if len(self.detection_list) == DETECTION_STABILITY_TIME and all(self.detection_list):
                    logger.info("Face detected consistently")
                    self.handle_detection(frame)
                    self.detection_list.clear()
                    logger.info(f"Waiting for {SLEEP_DURATION} seconds before next cycle")
                    time.sleep(SLEEP_DURATION)
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Program stopped by user")
        finally:
            self.stop_event.set()
            distance_thread.join(timeout=1)
            pzem_thread.join(timeout=1)
            cv2.destroyAllWindows()
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
    system = IntegratedSystem()
    system.run()

if __name__ == "__main__":
    main()