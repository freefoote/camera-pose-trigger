import logging
import threading
import time
import os
import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

def current_milli_time():
    return round(time.time() * 1000)

def thread_run_gesture_detector(event_emitter, camera_url, modelfile):
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=modelfile),
        running_mode=VisionRunningMode.VIDEO)

    with GestureRecognizer.create_from_options(options) as recognizer:
        # The landmarker is initialized. Use it here.
        logging.info("Opening capture...")
        framecount = 0
        vcap = cv2.VideoCapture(camera_url)
        logging.info("Capture has been opened.")
        keep_capturing = True

        @event_emitter.on('gesture_kill_thread')
        def kill_lm_thread():
            # Whee! Thread safety non existent!
            keep_capturing = False

        while keep_capturing:
            ret, frame = vcap.read()
            if not ret:
                logging.error("Failed to capture - aborting.")
                break

            if framecount % 10 == 0:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                frame_timestamp_ms = current_milli_time()
                gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)
                print(gesture_recognition_result)

                # event_emitter.emit('raw_frame', mp_image)
                # event_emitter.emit('guesture_annotated_frame', annotated_image)

            framecount += 1

        vcap.release()

if __name__ == "__main__":
    from dotenv import load_dotenv
    from pyee.base import EventEmitter

    load_dotenv()

    ee = EventEmitter()

    @ee.on('gesture_result')
    def lm_result_simple(foo):
        print(foo)

    @ee.on('gesture_annotated_frame')
    def lm_ann_frame(frame):
        cv2.imwrite('out.jpg', frame)

    x = threading.Thread(target=thread_run_gesture_detector, args=(ee, os.getenv('CAMERA'), os.getenv('GESTURE_MODEL_FILE')))
    logging.info("Starting capture thread...")
    x.start()