import time
import cv2
import logging
from datetime import datetime
import pygame
from dotenv import load_dotenv
import os

load_dotenv()

# Import sensor functionalities
from hc_sr04p_distance import filtered_distance, GPIO_setup, GPIO_cleanup
from Updated_PZEM_Sensor_Reader_Script import connect_to_sensor, read_sensor_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DISTANCE_THRESHOLD = 150.0  # cm
FACE_DETECTION_TIME = 10.0  # seconds
PZEM_MONITORING_TIME = 300.0  # 5 minutes in seconds
CURRENT_THRESHOLD = 1.0  # A
SLEEP_DURATION = 1.0  # 1 second between distance checks

class SensorManager:
    def __init__(self):
        self.camera = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.pzem_master = connect_to_sensor()
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
        GPIO_setup()

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

    def detect_face(self):
        face_detected_time = 0.0
        start_time = time.time()
        
        if not self.open_camera():
            logger.error("Failed to open camera for face detection")
            return False

        while time.time() - start_time < FACE_DETECTION_TIME:
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture frame")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                face_detected_time = time.time() - start_time
                logger.info(f"Face detected for {face_detected_time:.2f} seconds")
            else:
                face_detected_time = 0.0
                logger.info("No face detected")

            if face_detected_time >= FACE_DETECTION_TIME:
                self.close_camera()
                return True

        self.close_camera()
        return False

    def monitor_current(self):
        start_time = time.time()
        while time.time() - start_time < PZEM_MONITORING_TIME:
            pzem_data = read_sensor_data(self.pzem_master)
            try:
                current = float(pzem_data['current_A'])
                logger.info(f"Current reading: {current:.3f} A")
                if current > CURRENT_THRESHOLD:
                    logger.info(f"Current threshold exceeded: {current:.3f} A > {CURRENT_THRESHOLD:.3f} A")
                    return True
            except (ValueError, KeyError) as e:
                logger.error(f"Error processing PZEM data: {e}")
            time.sleep(1.0)
        logger.info(f"Current monitoring timed out after {PZEM_MONITORING_TIME:.0f} seconds")
        return False

    def play_audio(self, file_name):
        try:
            pygame.mixer.music.load(file_name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except pygame.error as e:
            logger.error(f"Error playing sound {file_name}: {e}")

    def run(self):
        while True:
            distance = filtered_distance()
            if distance is None:
                logger.warning("Failed to get distance reading")
                time.sleep(SLEEP_DURATION)
                continue

            try:
                distance = float(distance)
                logger.info(f"Measured distance: {distance:.2f} cm")

                if distance <= DISTANCE_THRESHOLD:
                    logger.info(f"Object detected within threshold: {distance:.2f} cm <= {DISTANCE_THRESHOLD:.2f} cm")
                    logger.info("Starting face detection")
                    if self.detect_face():
                        logger.info("Face detected for 10 seconds, starting current monitoring")
                        if self.monitor_current():
                            logger.info("Current above threshold detected, playing do_nothing.mp3")
                            self.play_audio("do_nothing.mp3")
                        else:
                            logger.info("No significant current detected, playing alert.mp3")
                            self.play_audio("alert.mp3")
                        break
                    else:
                        logger.info("Face not detected for 10 seconds, continuing distance monitoring")
            except ValueError:
                logger.error(f"Invalid distance value: {distance}")
            
            time.sleep(SLEEP_DURATION)

    def cleanup(self):
        self.close_camera()
        self.pzem_master.close()
        GPIO_cleanup()

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