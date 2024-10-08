import time
import cv2
from ultralytics import YOLO
import logging
from datetime import datetime
import pygame
from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFS
import os
import threading
from queue import Queue
from collections import deque

# Assume these imports are correctly set up in your environment
from hc_sr04p_distance import filtered_distance
from Updated_PZEM_Sensor_Reader_Script import connect_to_sensor, read_sensor_data
from line_notify import LineNotifier

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DISTANCE_THRESHOLD = 150  # cm
CHARGING_TIMEOUT = 60  # 1 minute in seconds
CURRENT_THRESHOLD = 1.0  # A
DETECTION_STABILITY_TIME = 5  # seconds
DISTANCE_READINGS = 5  # Number of distance readings to average
SLEEP_DURATION = 10  # 10 seconds between test cycles

# MongoDB setup
MONGO_URI = "mongodb+srv://Extremenop:Nop24681036@cardb.ynz57.mongodb.net/?retryWrites=true&w=majority&appName=Cardb"
DB_NAME = "cardb"
COLLECTION_NAME = "nonev"

# LINE Notify Token
LINE_NOTIFY_TOKEN = "J0oQ74OftbCNdiPCCfV4gs75aqtz4aAL8NiGfHERvZ4"

class SensorManager:
    def __init__(self):
        self.camera = None
        self.model = YOLO('yolov9s_ev.pt')
        self.pzem_master = connect_to_sensor()
        self.line_notifier = LineNotifier()
        self.line_notifier.token = LINE_NOTIFY_TOKEN
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
        # MongoDB setup
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.fs = GridFS(self.db)

        # Threading setup
        self.distance_queue = Queue()
        self.pzem_queue = Queue()
        self.detection_queue = Queue()
        self.stop_event = threading.Event()

        # Deques for storing recent readings
        self.distance_deque = deque(maxlen=DISTANCE_READINGS)
        self.detection_deque = deque(maxlen=DETECTION_STABILITY_TIME)

    def distance_thread(self):
        while not self.stop_event.is_set():
            distance = filtered_distance()
            if distance is not None:
                self.distance_queue.put(distance)
            time.sleep(1)

    def pzem_thread(self):
        while not self.stop_event.is_set():
            pzem_data = read_sensor_data(self.pzem_master)
            self.pzem_queue.put(pzem_data)
            time.sleep(1)

    def detection_thread(self):
        while not self.stop_event.is_set():
            if self.camera is None or not self.camera.isOpened():
                if not self.open_camera():
                    time.sleep(1)
                    continue
            
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture frame")
                self.close_camera()
                time.sleep(1)
                continue
            
            results = self.model(frame)
            detection = self.process_yolo_results(results)
            self.detection_queue.put((detection, frame))
            time.sleep(0.1)

    def process_yolo_results(self, results):
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 2 or cls == 80:  # Assuming class 2 is 'face' for testing
                    return True
        return False

    def open_camera(self):
        try:
            if self.camera is None:
                self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Camera failed to open")
                return False
            logger.info("Camera opened successfully")
            return True
        except Exception as e:
            logger.error(f"Error opening camera: {str(e)}")
            return False

    def close_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            logger.info("Camera closed")

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
        return self.fs.put(img_encoded.tobytes(), filename=f"vehicle_{timestamp}.jpg", metadata={"timestamp": timestamp})

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
        message = f"{timestamp} Face detected and not charging"
        
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
        threads = [
            threading.Thread(target=self.distance_thread),
            threading.Thread(target=self.pzem_thread),
            threading.Thread(target=self.detection_thread)
        ]
        for thread in threads:
            thread.start()

        try:
            while not self.stop_event.is_set():
                # Process distance readings
                while not self.distance_queue.empty():
                    distance = self.distance_queue.get()
                    self.distance_deque.append(distance)

                # Process detections
                while not self.detection_queue.empty():
                    detection, frame = self.detection_queue.get()
                    self.detection_deque.append(detection)

                # Check conditions
                if len(self.distance_deque) == DISTANCE_READINGS and len(self.detection_deque) == DETECTION_STABILITY_TIME:
                    avg_distance = sum(self.distance_deque) / len(self.distance_deque)
                    all_detected = all(self.detection_deque)

                    if avg_distance <= DISTANCE_THRESHOLD and all_detected:
                        logger.info("Face detected consistently and within range")
                        self.handle_detection(frame)

                        # Monitor PZEM for 1 minute
                        start_time = time.time()
                        charging_detected = False
                        while time.time() - start_time < CHARGING_TIMEOUT:
                            if not self.pzem_queue.empty():
                                pzem_data = self.pzem_queue.get()
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

        except KeyboardInterrupt:
            logger.info("Program stopped by user")
        finally:
            self.stop_event.set()
            for thread in threads:
                thread.join()
            self.cleanup()

    def cleanup(self):
        self.close_camera()
        self.pzem_master.close()
        self.client.close()
        logger.info("Cleanup completed")

def main():
    manager = SensorManager()
    manager.run()

if __name__ == "__main__":
    main()