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

from simpleeval import SimpleEval

import cv2
import numpy as np
import base64

import posedetector

# Fire up the landmarker and event emitter.
event_emitter = EventEmitter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("Starting LM thread...")
    pose_thread = threading.Thread(target=posedetector.thread_run_pose_detector, args=(event_emitter, os.getenv('CAMERA'), os.getenv('POSE_MODEL_FILE')))
    pose_thread.start()
    print("LM Thread running, yiedling back.")
    yield
    # Clean up the ML models and release the resources
    print("Lifecycle complete. Closing thread.")
    event_emitter.emit('pose_kill_thread')

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

    # TODO: This probably isn't thread safe...
    @event_emitter.on('pose_annotated_frame')
    def lm_ann_frame(frame):
        image_string = convert_image_to_base64(frame)
        asyncio.run(websocket.send_text('image:' + image_string))
    @event_emitter.on('pose_result')
    def lm_result(meta):
        asyncio.run(websocket.send_text('raw:' + json.dumps(meta)))

    while True:
        data = await websocket.receive_text()
        print("Got from websocket:", data)

# Evaluate the result against the rules.
rules = [
    {
        'name': 'Left Arm Up',
        'expression': "(left_arm_whole_angle < -150 and right_arm_whole_octant > -150)",
        'event': 'pause'
    },
    {
        'name': 'Right Arm Up',
        'expression': "(right_arm_whole_angle < -150 and left_arm_whole_octant > -150)",
        'event': 'resume'
    },
]

evaluator = SimpleEval()
parsed_rules = {}
for rule in rules:
    key = rule['name']
    parsed = evaluator.parse(rule['expression'])
    parsed_rules[key] = {
        'parsed': parsed,
        'expression': rule['expression']
    }

@event_emitter.on('pose_result')
def lm_result(meta):
    for person in meta['persons']:
        # Check all rules against the person.
        for rule_name in parsed_rules:
            rule_data = parsed_rules[rule_name]
            evaluator.names = person
            result = evaluator.eval(rule_data['expression'], previously_parsed=rule_data['parsed'])
            if result:
                print("Expression", rule_name, "(", rule_data['expression'], ")", "matched.")
