# Python standard libraries.
import logging
import time
import base64

# Third party libraries.
import numpy
import cv2

def default_logging_config():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

def subprocess_emit(message_type, content):
    payload = { 'message_type': message_type, 'content': content }
    print(json.dumps(paylod), flush=True)

def current_milli_time():
    return round(time.time() * 1000)

def convert_image_to_base64(frame: numpy.ndarray) -> bytes:
    _, encoded_image = cv2.imencode('.jpg', frame)
    image_bytes = encoded_image.tobytes()
    image_string = 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode('utf-8')
    return image_string