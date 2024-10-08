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

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DETECTION_STABILITY_TIME = 5  # seconds
SLEEP_DURATION = 10  # 10 seconds between test cycles

# MongoDB setup
LINE_NOTIFY_TOKEN = "J0oQ74OftbCNdiPCCfV4gs75aqtz4aAL8NiGfHERvZ4"
MONGO_URI = "mongodb+srv://Extremenop:Nop24681036@cardb.ynz57.mongodb.net/?retryWrites=true&w=majority&appName=Cardb"
DB_NAME = "cardb"
COLLECTION_NAME = "nonev"



class FaceDetectionSystem:
    def __init__(self):
        self.camera = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.line_notifier = None
        self.client = None
        self.db = None
        self.collection = None
        self.fs = None
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
        # List for storing recent detections
        self.detection_list = []

    def initialize_camera(self):
        print("Initializing camera...")
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Failed to open camera")
            return False
        print("Camera initialized successfully")
        return True

    def initialize_line(self):
        try:
            self.line_notifier = LineNotifier()  # Remove the argument
            self.line_notifier.token = LINE_NOTIFY_TOKEN  # Set the token separately
            print("LINE Notifier initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize LINE Notifier: {e}")
            return False

    def initialize_mongodb(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            self.fs = GridFS(self.db)
            # Test the connection
            self.client.server_info()
            print("MongoDB initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize MongoDB: {e}")
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
            print(f"Error playing sound {sound_file}: {e}")

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
        
        print("Playing alert sound")
        self.play_sound("alert.mp3")
        
        local_path = f"detected_face_{timestamp.replace(':', '-')}.jpg"
        cv2.imwrite(local_path, frame)
        
        try:
            file_id = self.save_image_to_gridfs(frame, timestamp)
            metadata_id = self.save_metadata(file_id, timestamp, event)
            print(f"Image saved to GridFS with ID: {file_id}")
            print(f"Metadata saved with ID: {metadata_id.inserted_id}")
        except Exception as e:
            print(f"Failed to save image or metadata: {e}")
        
        try:
            self.line_notifier.send_image(message, local_path)
            print("Line notification sent successfully")
        except Exception as e:
            print(f"Failed to send Line notification: {e}")

    def run(self):
        if not self.initialize_services():
            print("Failed to initialize services. Exiting.")
            return

        print("Starting main detection loop")
        while True:
            if not self.camera.isOpened():
                print("Camera is not open. Attempting to reopen...")
                if not self.initialize_camera():
                    print("Failed to reopen camera. Retrying in 5 seconds.")
                    time.sleep(5)
                    continue

            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame from camera")
                time.sleep(1)
                continue

            face_detected = self.detect_face(frame)
            print(f"Face detected: {face_detected}")

            # Draw a rectangle around the face if detected
            if face_detected:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.detection_list.append(face_detected)
            if len(self.detection_list) > DETECTION_STABILITY_TIME:
                self.detection_list.pop(0)

            if len(self.detection_list) == DETECTION_STABILITY_TIME and all(self.detection_list):
                print("Face detected consistently")
                self.handle_detection(frame)

                # Clear detection list for next cycle
                self.detection_list.clear()

                # Wait for 10 seconds before next cycle
                print(f"Waiting for {SLEEP_DURATION} seconds before next cycle")
                time.sleep(SLEEP_DURATION)
            
            time.sleep(0.1)  # Small delay to prevent CPU overuse

        cv2.destroyAllWindows()

    def cleanup(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        if self.client:
            self.client.close()
        print("Cleanup completed")

def main():
    system = FaceDetectionSystem()
    try:
        system.run()
    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()