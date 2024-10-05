import Jetson.GPIO as GPIO
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TRIG_PIN = 12  # Physical Pin 12
ECHO_PIN = 16  # Physical Pin 16
MIN_DISTANCE = 2  # cm
MAX_DISTANCE = 400  # cm
TIMEOUT = 1.0  # seconds
MEASUREMENT_INTERVAL = 1  # seconds

# Set the GPIO mode to BOARD (physical pin numbering)
GPIO.setmode(GPIO.BOARD)

# Set up GPIO pins
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def get_distance():
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
        logging.warning(f"Distance out of range: {distance} cm")
        return None
    
    return round(distance, 2)

last_valid_distance = None

def filtered_distance():
    global last_valid_distance
    for _ in range(3):
        dist = get_distance()
        if dist is not None:
            if last_valid_distance is None or abs(dist - last_valid_distance) < 50:
                last_valid_distance = dist
                return dist
    logging.warning("Failed to get a valid distance after 3 attempts")
    return None

try:
    while True:
        dist = filtered_distance()
        if dist is None:
            logging.error("Measurement error or out of range")
        else:
            logging.info(f"Measured Distance = {dist} cm")
        time.sleep(MEASUREMENT_INTERVAL)
except KeyboardInterrupt:
    logging.info("Measurement stopped by user")
finally:
    GPIO.cleanup()
    logging.info("GPIO cleanup completed")