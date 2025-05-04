# Python standard libraries.
import logging
import time
import base64
import json

# Third party libraries.
import numpy
import cv2

def default_logging_config():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

def subprocess_emit(message_type, content):
    payload = { 'message_type': message_type, 'content': content }
    print(json.dumps(payload), flush=True)

def current_milli_time():
    return round(time.time() * 1000)

def convert_image_to_base64(frame: numpy.ndarray) -> bytes:
    resized = resize_image_to_max_width(frame, 800)
    _, encoded_image = cv2.imencode('.jpg', resized)
    image_bytes = encoded_image.tobytes()
    image_string = 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode('utf-8')
    return image_string

def resize_image_to_max_width(image, max_width):
    height, width, _ = image.shape
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        resized_image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    else:
      resized_image = image
    return resized_image