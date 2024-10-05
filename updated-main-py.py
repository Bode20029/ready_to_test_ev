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
CHARGING_TIMEOUT = 600  # 10 minutes in seconds
CURRENT_THRESHOLD = 0.1  # A
DETECTION_STABILITY_TIME = 5  # seconds
DISTANCE_READINGS = 2  # Number of distance readings to average
SLEEP_DURATION = 120  # 2 minutes in seconds

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

    def check_distance(self):
        distances = []
        for _ in range(DISTANCE_READINGS):
            reading = filtered_distance()
            if reading is not None:
                try:
                    # Extract the numeric part and convert to float
                    numeric_distance = float(str(reading).split()[0])
                    distances.append(numeric_distance)
                except (ValueError, IndexError):
                    logger.warning(f"Invalid distance reading: {reading}")
        
        avg_distance = sum(distances) / len(distances) if distances else None
        if avg_distance is not None:
            logger.info(f"Measured Distance = {avg_distance:.2f} cm")
        return avg_distance

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

    def detect_vehicle(self):
        detection_start_time = time.time()
        logger.info("Entering detect_vehicle method")
        
        if not self.open_camera():
            logger.error("Failed to open camera in detect_vehicle")
            return None, None
        
        while time.time() - detection_start_time < DETECTION_STABILITY_TIME:
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture frame")
                self.close_camera()
                return None, None
            
            results = self.model(frame)
            car_detected, is_ev = self.process_yolo_results(results)
            
            if car_detected:
                logger.info(f"Vehicle detected: {'EV' if is_ev else 'Non-EV'}")
                return is_ev, frame
        
        logger.info("No vehicle detected within stability time")
        self.close_camera()
        return None, None

    def process_yolo_results(self, results):
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 2:  # Assuming class 2 is 'car'
                    return True, False
                elif cls == 80:  # Assuming class 80 is 'ev'
                    return True, True
        return False, False

    def monitor_charging(self):
        start_time = time.time()
        while time.time() - start_time < CHARGING_TIMEOUT:
            distance = self.check_distance()
            if distance is None or distance > DISTANCE_THRESHOLD:
                logger.info("Vehicle left before charging timeout")
                return None  # Vehicle left before timeout
            
            pzem_data = read_sensor_data(self.pzem_master)
            if pzem_data['current_A'] > CURRENT_THRESHOLD:
                return True  # Charging detected
            time.sleep(5)
        return False  # Timeout reached, no charging detected

    def play_alert(self):
        for sound_file in ["alert.mp3", "Warning.mp3"]:
            try:
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play()
                pygame.time.wait(int(pygame.mixer.Sound(sound_file).get_length() * 1000))
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

    def handle_detection(self, frame, is_ev):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event = "non_charging_ev" if is_ev else "non_ev_parked"
        message = f"{timestamp} {'an Electric vehicle is not charging' if is_ev else 'a non-EV car is detected'}"
        
        self.play_alert()
        
        local_path = f"detected_vehicle_{timestamp.replace(':', '-')}.jpg"
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
        while True:
            distance = self.check_distance()
            if distance is None:
                logger.warning("Failed to get valid distance measurement")
                time.sleep(1)
                continue
            
            if distance > DISTANCE_THRESHOLD:
                logger.info(f"No object within threshold. Distance: {distance:.2f} cm")
                time.sleep(1)
                continue

            logger.info(f"Object detected within threshold. Distance: {distance:.2f} cm")
            is_ev, frame = self.detect_vehicle()
            if is_ev is None:
                logger.info("No vehicle consistently detected, continuing to monitor")
                continue
            elif not is_ev:
                logger.info("Non-EV detected, handling detection")
                self.handle_detection(frame, is_ev=False)
            else:
                logger.info("EV detected, monitoring charging")
                charging_result = self.monitor_charging()
                if charging_result is None:
                    logger.info("Vehicle left during charging monitoring")
                    continue
                elif not charging_result:
                    logger.info("EV not charging, handling detection")
                    self.handle_detection(frame, is_ev=True)
                else:
                    logger.info("EV charged successfully")
            
            self.close_camera()
            time.sleep(SLEEP_DURATION)

    def cleanup(self):
        self.close_camera()
        self.pzem_master.close()
        self.client.close()

def main():
    manager = SensorManager()
    try:
        manager.run()
    except KeyboardInterrupt:
        logger.info("Program stopped by user")
    finally:
        manager.cleanup()
        logger.info("Cleanup completed")

if __name__ == "__main__":
    main()