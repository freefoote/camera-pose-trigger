from dotenv import load_dotenv

load_dotenv()

import logging
import threading
import time
import os

from pyee.base import EventEmitter

from fastapi import FastAPI, WebSocket, Response
from contextlib import asynccontextmanager
from nicegui import app as ngapp, ui, run

import cv2
import numpy as np
import base64

import landmarker

# Fire up the landmarker and event emitter.
event_emitter = EventEmitter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("Starting LM thread...")
    lm_thread = threading.Thread(target=landmarker.thread_run_mediapipe, args=(event_emitter, os.getenv('CAMERA'), os.getenv('MODELFILE')))
    lm_thread.start()
    print("LM Thread running, yiedling back.")
    yield
    # Clean up the ML models and release the resources
    print("Lifecycle complete. Closing thread.")
    event_emitter.emit('kill_lm_thread')

# Set up FastAPI
app = FastAPI(lifespan=lifespan)

# The encoded frame to send back to the client.
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
current_frame_encoded_as_response = [None]
current_frame_encoded_as_response[0] = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')

def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()

@event_emitter.on('annotated_frame')
def lm_ann_frame(frame):
    # Encode the image to a memory buffer
    jpeg = convert(frame)
    current_frame_encoded_as_response[0] = Response(content=jpeg, media_type='image/jpeg')

@app.get('/video/frame')
async def grab_video_frame() -> Response:
    return current_frame_encoded_as_response[0]

@ui.page('/')
def show():
    ui.markdown('# Pose Trigger')

    with ui.card().tight():
        video_image = ui.interactive_image().classes('w-full h-full')
        ui.timer(interval=0.2, callback=lambda: video_image.set_source(f'/video/frame?{time.time()}'))
        with ui.card_section():
            ui.label('Lorem ipsum dolor sit amet, consectetur adipiscing elit, ...')

ui.run_with(app)