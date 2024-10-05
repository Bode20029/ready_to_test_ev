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
import requests

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['ULTRALYTICS_HOME'] = os.path.expanduser('~/ultralytics_cache')

# Constants
DETECTION_STABILITY_TIME = 5  # seconds
SLEEP_DURATION = 120  # 2 minutes in seconds

# MongoDB setup
MONGO_URI = "mongodb+srv://Extremenop:Nop24681036@cardb.ynz57.mongodb.net/?retryWrites=true&w=majority&appName=Cardb"
DB_NAME = "cardb"
COLLECTION_NAME = "nonev"

# LINE Notify Token
LINE_NOTIFY_TOKEN = "J0oQ74OftbCNdiPCCfV4gs75aqtz4aAL8NiGfHERvZ4"

class LineNotifier:
    def __init__(self):
        self.token = None

    def send_image(self, message, image_path):
        if not self.token:
            raise ValueError("LINE Notify token is not set")

        url = 'https://notify-api.line.me/api/notify'
        headers = {'Authorization': f'Bearer {self.token}'}
        data = {'message': message}
        files = {'imageFile': open(image_path, 'rb')}
        
        response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()

class SensorManager:
    def __init__(self):
        self.camera = None
        self.model = YOLO('YoloV9_Car.pt')
        self.line_notifier = LineNotifier()
        self.line_notifier.token = LINE_NOTIFY_TOKEN
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
        # MongoDB setup
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.fs = GridFS(self.db)

    def open_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
        return self.camera.isOpened()

    def close_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def detect_vehicle(self):
        detection_start_time = time.time()
        while time.time() - detection_start_time < DETECTION_STABILITY_TIME:
            if not self.open_camera():
                logger.error("Failed to open camera")
                return None, None
            
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture frame")
                return None, None

            results = self.model(frame)
            car_detected, is_ev = self.process_yolo_results(results)
            
            if car_detected:
                return is_ev, frame

        return None, None

    def process_yolo_results(self, results):
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Assuming class 0 is 'non-EV'
                    return True, False
                elif cls == 1:  # Assuming class 1 is 'EV'
                    return True, True
        return False, False

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
        event = "ev_detected" if is_ev else "non_ev_detected"
        message = f"{timestamp} {'An Electric vehicle is detected' if is_ev else 'A non-EV car is detected'}"
        
        # Play alert
        self.play_alert()
        
        # Save locally
        local_path = f"detected_vehicle_{timestamp.replace(':', '-')}.jpg"
        cv2.imwrite(local_path, frame)
        
        # Save to GridFS and metadata collection
        try:
            file_id = self.save_image_to_gridfs(frame, timestamp)
            metadata_id = self.save_metadata(file_id, timestamp, event)
            logger.info(f"Image saved to GridFS with ID: {file_id}")
            logger.info(f"Metadata saved with ID: {metadata_id.inserted_id}")
        except Exception as e:
            logger.error(f"Failed to save image or metadata: {e}")
        
        # Send Line notification
        try:
            self.line_notifier.send_image(message, local_path)
            logger.info("Line notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Line notification: {e}")

    def run(self):
        is_ev, frame = self.detect_vehicle()
        if is_ev is None:
            logger.info("No vehicle detected")
        else:
            self.handle_detection(frame, is_ev)

        self.close_camera()

    def cleanup(self):
        self.close_camera()
        self.client.close()

def main():
    manager = SensorManager()
    try:
        while True:
            manager.run()
            logger.info(f"Sleeping for {SLEEP_DURATION} seconds before next cycle")
            time.sleep(SLEEP_DURATION)
    except KeyboardInterrupt:
        logger.info("Program stopped by user")
    finally:
        manager.cleanup()
        logger.info("Cleanup completed")

if __name__ == "__main__":
    main()