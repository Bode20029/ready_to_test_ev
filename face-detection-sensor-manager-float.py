import cv2
import time
import logging
import threading
from queue import Queue
from collections import deque
import numpy as np
import pygame
from pymongo import MongoClient
from gridfs import GridFS
from line_notify import LineNotifier
from dotenv import load_dotenv
import os
from datetime import datetime

# Assume these imports are correctly set up in your environment
from hc_sr04p_distance import filtered_distance
from Updated_PZEM_Sensor_Reader_Script import connect_to_sensor, read_sensor_data

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DISTANCE_THRESHOLD = 150  # cm
CHARGING_TIMEOUT = 60  # 1 minute in seconds
CURRENT_THRESHOLD = 1.0  # A
DETECTION_STABILITY_TIME = 5  # seconds
DISTANCE_READINGS = 5  # Number of distance readings to average
SLEEP_DURATION = 10  # 10 seconds between test cycles

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

# LINE Notify Token
LINE_NOTIFY_TOKEN = os.getenv('LINE_NOTIFY_TOKEN')

class SensorManager:
    def __init__(self):
        self.camera = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.pzem_master = None
        self.line_notifier = None
        self.client = None
        self.db = None
        self.collection = None
        self.fs = None
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
        # Deques for storing recent readings
        self.distance_deque = deque(maxlen=DISTANCE_READINGS)
        self.detection_deque = deque(maxlen=DETECTION_STABILITY_TIME)

        # Threading setup
        self.stop_event = threading.Event()

    def initialize_camera(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False
            logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False

    def initialize_line(self):
        try:
            self.line_notifier = LineNotifier(LINE_NOTIFY_TOKEN)
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
            # Test the connection
            self.client.server_info()
            logger.info("MongoDB initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            return False

    def initialize_pzem(self):
        try:
            self.pzem_master = connect_to_sensor()
            logger.info("PZEM sensor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PZEM sensor: {e}")
            return False

    def initialize_services(self):
        if not self.initialize_camera():
            logger.error("Failed to initialize camera. Exiting.")
            return False
        
        if not self.initialize_line():
            logger.error("Failed to initialize LINE Notifier. Exiting.")
            return False
        
        if not self.initialize_mongodb():
            logger.error("Failed to initialize MongoDB. Exiting.")
            return False
        
        if not self.initialize_pzem():
            logger.error("Failed to initialize PZEM sensor. Exiting.")
            return False
        
        logger.info("All services initialized successfully")
        return True

    def check_sensors(self):
        if not self.camera.isOpened():
            logger.error("Camera is not open")
            return False
        
        # Test camera by capturing a frame
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to capture frame from camera")
            return False
        
        logger.debug("Camera is functioning properly")
        
        distance = filtered_distance()
        if distance is None:
            logger.error("Failed to get distance reading")
            return False
        
        logger.debug(f"Distance sensor reading: {distance} cm")
        return True

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

    def run(self):
        if not self.initialize_services():
            logger.error("Failed to initialize all services. Exiting.")
            return

        logger.info("Starting main detection loop")
        while not self.stop_event.is_set():
            if not self.check_sensors():
                logger.error("Sensors not functioning properly. Retrying in 5 seconds.")
                time.sleep(5)
                continue

            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture frame")
                continue

            distance = filtered_distance()
            face_detected = self.detect_face(frame)

            logger.debug(f"Distance: {distance} cm, Face detected: {face_detected}")

            self.distance_deque.append(distance)
            self.detection_deque.append(face_detected)

            if len(self.distance_deque) == DISTANCE_READINGS and len(self.detection_deque) == DETECTION_STABILITY_TIME:
                avg_distance = np.mean(self.distance_deque)
                all_detected = all(self.detection_deque)

                if avg_distance <= DISTANCE_THRESHOLD and all_detected:
                    logger.info(f"Face detected consistently and within range. Avg distance: {avg_distance:.2f} cm")
                    self.handle_detection(frame)

                    # Monitor PZEM for 1 minute
                    start_time = time.time()
                    charging_detected = False
                    while time.time() - start_time < CHARGING_TIMEOUT:
                        pzem_data = read_sensor_data(self.pzem_master)
                        if pzem_data['current_A'] > CURRENT_THRESHOLD:
                            charging_detected = True
                            break
                        time.sleep(1)

                    if not charging_detected:
                        logger.info("Not charging after 1 minute")
                        self.play_sound("not_charging.mp3")

                    # Clear deques for next cycle
                    self.distance_deque.clear()
                    self.detection_deque.clear()

                    # Wait for 10 seconds before next test cycle
                    time.sleep(SLEEP_DURATION)

            time.sleep(0.1)  # Small sleep to prevent CPU hogging

    def cleanup(self):
        if self.camera:
            self.camera.release()
        if self.pzem_master:
            self.pzem_master.close()
        if self.client:
            self.client.close()
        logger.info("Cleanup completed")

def main():
    manager = SensorManager()
    try:
        manager.run()
    except KeyboardInterrupt:
        logger.info("Program stopped by user")
    finally:
        manager.cleanup()

if __name__ == "__main__":
    main()