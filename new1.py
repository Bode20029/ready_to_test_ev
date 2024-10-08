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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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

        # Initialize pygame for audio
        pygame.mixer.init()

        # Initialize GPIO for distance sensor
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)

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
            print("Attempting to create LineNotifier instance")
            self.line_notifier = LineNotifier()
            print("LineNotifier instance created")
            print("Setting token: {}".format(LINE_NOTIFY_TOKEN))
            self.line_notifier.token = LINE_NOTIFY_TOKEN
            print("LINE Notifier initialized successfully")
            return True
        except Exception as e:
            print("Failed to initialize LINE Notifier: {}".format(str(e)))
            print("Error type: {}".format(type(e).__name__))
            import traceback
            traceback.print_exc()
            return False

    def initialize_mongodb(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            self.fs = GridFS(self.db)
            self.client.server_info()
            print("MongoDB initialized successfully")
            return True
        except Exception as e:
            print("Failed to initialize MongoDB: {}".format(str(e)))
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
            print("Error playing sound {}: {}".format(sound_file, e))

    def save_image_to_gridfs(self, frame, timestamp):
        _, img_encoded = cv2.imencode('.jpg', frame)
        return self.fs.put(img_encoded.tobytes(), filename="face_{}.jpg".format(timestamp), metadata={"timestamp": timestamp})

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
        message = "{} Face detected".format(timestamp)

        print("Playing alert sound")
        self.play_sound("alert.mp3")

        local_path = "detected_face_{}.jpg".format(timestamp.replace(':', '-'))

        cv2.imwrite(local_path, frame)

        try:
            file_id = self.save_image_to_gridfs(frame, timestamp)
            metadata_id = self.save_metadata(file_id, timestamp, event)
            print("Image saved to GridFS with ID: {}".format(file_id))
            print("Metadata saved with ID: {}".format(metadata_id.inserted_id))
        except Exception as e:
            print("Failed to save image or metadata: {}".format(e))

        try:
            self.line_notifier.send_image(message, local_path)
            print("Line notification sent successfully")
        except Exception as e:
            print("Failed to send Line notification: {}".format(e))

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
                logging.warning("Echo pulse start timeout")
                return None

        while GPIO.input(ECHO_PIN) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end > timeout:
                logging.warning("Echo pulse end timeout")
                return None

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150

        if distance < MIN_DISTANCE or distance > MAX_DISTANCE:
            logging.warning("Distance out of range: {} cm".format(distance))
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
                logging.warning("Failed to get a valid distance after 3 attempts")

            # Adaptive scheduling
            if self.vehicle_present:
                time.sleep(DISTANCE_MEASUREMENT_INTERVAL / 2)  # More frequent readings when vehicle present
            else:
                time.sleep(DISTANCE_MEASUREMENT_INTERVAL)

    def pzem_sensor_thread(self):
        master = self.connect_to_pzem()
        while not self.stop_event.is_set():
            try:
                data = self.read_pzem_data(master)
                self.pzem_sensor_data.put(data)
            except Exception as e:
                logging.error("Error reading PZEM data: {}".format(e))
                master.close()
                master = self.connect_to_pzem()
            
            # Adaptive scheduling
            if self.vehicle_present:
                time.sleep(PZEM_MEASUREMENT_INTERVAL / 4)  # More frequent readings when vehicle present
            else:
                time.sleep(PZEM_MEASUREMENT_INTERVAL)

    def run(self):
        if not self.initialize_services():
            print("Failed to initialize services. Exiting.")
            return

        # Start sensor threads
        distance_thread = threading.Thread(target=self.distance_sensor_thread)
        pzem_thread = threading.Thread(target=self.pzem_sensor_thread)
        distance_thread.start()
        pzem_thread.start()

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
            print("Face detected: {}".format(face_detected))

            # Draw a rectangle around the face if detected
            if face_detected:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Process sensor data
            try:
                distance = self.distance_sensor_data.get_nowait()
                if distance < 100:  # Adjust this threshold as needed
                    self.vehicle_present = True
                else:
                    self.vehicle_present = False
                print("Distance: {} cm".format(distance))
            except queue.Empty:
                pass

            try:
                pzem_data = self.pzem_sensor_data.get_nowait()
                print("PZEM Data: {}".format(json.dumps(pzem_data, indent=2)))
            except queue.Empty:
                pass

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
                print("Waiting for {} seconds before next cycle".format(SLEEP_DURATION))
                time.sleep(SLEEP_DURATION)
            
            time.sleep(0.1)  # Small delay to prevent CPU overuse

        self.stop_event.set()
        distance_thread.join()
        pzem_thread.join()
        cv2.destroyAllWindows()

    def cleanup(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        if self.client:
            self.client.close()
        GPIO.cleanup()
        print("Cleanup completed")

def main():
    system = IntegratedSystem()
    try:
        system.run()
    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()