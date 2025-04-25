from dotenv import load_dotenv

load_dotenv()

import logging
import threading
import time
import os
import asyncio
import json

from pyee.base import EventEmitter

from fastapi import FastAPI, WebSocket, Response
from contextlib import asynccontextmanager

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

def convert_image_to_base64(frame: np.ndarray) -> bytes:
    _, encoded_image = cv2.imencode('.jpg', frame)
    image_bytes = encoded_image.tobytes()
    image_string = 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode('utf-8')
    return image_string

@app.get('/')
async def root() -> Response:
    with open('frontend.html', 'r') as file:
        return Response(content=file.read(), media_type="text/html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    @event_emitter.on('annotated_frame')
    def lm_ann_frame(frame):
        image_string = convert_image_to_base64(frame)
        asyncio.run(websocket.send_text('image:' + image_string))
    @event_emitter.on('landmarker_result')
    def lm_ann_frame(meta):
        asyncio.run(websocket.send_text('raw:' + json.dumps(meta)))
    while True:
        data = await websocket.receive_text()
        print("Got from websocket:", data)