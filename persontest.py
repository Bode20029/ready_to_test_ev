import cv2
import pygame
import os
import time
from pymongo import MongoClient
from gridfs import GridFS
import datetime
from line_notify import LineNotifier
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hardcoded MongoDB setup
MONGO_URI = "mongodb+srv://Extremenop:Nop24681036@cardb.ynz57.mongodb.net/?retryWrites=true&w=majority&appName=Cardb"
DB_NAME = "cardb"
COLLECTION_NAME = "nonev"

# Hardcoded Line Notify Token
LINE_NOTIFY_TOKEN = "J0oQ74OftbCNdiPCCfV4gs75aqtz4aAL8NiGfHERvZ4"

try:
    # MongoDB connection
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    fs = GridFS(db)
    logging.info("Successfully connected to MongoDB")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    raise

# Line Notify setup
line_notifier = LineNotifier()
line_notifier.token = LINE_NOTIFY_TOKEN

# Initialize pygame for audio
pygame.mixer.init()

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def play_alert():
    for sound_file in ["alert.mp3", "Warning.mp3"]:
        try:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            pygame.time.wait(int(pygame.mixer.Sound(sound_file).get_length() * 1000))
        except pygame.error as e:
            logging.error(f"Error playing sound {sound_file}: {e}")

def save_image_to_gridfs(frame, timestamp):
    _, img_encoded = cv2.imencode('.jpg', frame)
    return fs.put(img_encoded.tobytes(), filename=f"face_{timestamp}.jpg", metadata={"timestamp": timestamp})

def save_metadata(file_id, timestamp):
    metadata = {
        "file_id": file_id,
        "timestamp": timestamp,
        "event": "face_detected"
    }
    return collection.insert_one(metadata)

def handle_detection(frame):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Play alert
    play_alert()
    
    # Save locally
    local_path = f"detected_face_{timestamp.replace(':', '-')}.jpg"
    cv2.imwrite(local_path, frame)
    
    # Save to GridFS and metadata collection
    try:
        file_id = save_image_to_gridfs(frame, timestamp)
        metadata_id = save_metadata(file_id, timestamp)
        logging.info(f"Image saved to GridFS with ID: {file_id}")
        logging.info(f"Metadata saved with ID: {metadata_id.inserted_id}")
    except Exception as e:
        logging.error(f"Failed to save image or metadata: {e}")
    
    # Send Line notification
    try:
        message = f"A face is detected at {timestamp}"
        line_notifier.send_image(message, local_path)
        logging.info("Line notification sent successfully")
    except Exception as e:
        logging.error(f"Failed to send Line notification: {e}")
    
    # Wait for 3 minutes before restarting camera
    time.sleep(180)

def main():
    while True:
        cap = cv2.VideoCapture(0)
        face_detected = False
        counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                if not face_detected:
                    face_detected = True
                    counter = 0
                counter += 1
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                if counter == 10:
                    cap.release()
                    cv2.destroyAllWindows()
                    handle_detection(frame)
                    break
            else:
                face_detected = False
                counter = 0

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                client.close()
                return

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    client.close()

if __name__ == "__main__":
    main()