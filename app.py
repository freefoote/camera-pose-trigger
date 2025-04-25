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
import urllib

import posedetector

from ha_mqtt_discoverable import Settings, DeviceInfo
from ha_mqtt_discoverable.sensors import BinarySensor, BinarySensorInfo
from slugify import slugify

def configured_slugify(input):
    return slugify(input, separator='_')

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
        'expression': "(left_arm_whole_angle < -150 and right_arm_whole_octant > -150)"
    },
    {
        'name': 'Right Arm Up',
        'expression': "(right_arm_whole_angle < -150 and left_arm_whole_octant > -150)"
    },
]

mqtt_parsed = urllib.parse.urlparse(os.getenv("HA_MQTT"))
mqtt_settings = Settings.MQTT(host=mqtt_parsed.hostname, port=mqtt_parsed.port, username=mqtt_parsed.username, password=mqtt_parsed.password)
HA_DEVICE_NAME = os.getenv("HA_DEVICE_NAME")
ha_device_info = DeviceInfo(name=HA_DEVICE_NAME, identifiers=configured_slugify(HA_DEVICE_NAME))

evaluator = SimpleEval()
parsed_rules = {}
for rule in rules:
    key = rule['name']
    parsed = evaluator.parse(rule['expression'])

    motion_sensor_info = BinarySensorInfo(name=rule['name'], device_class='motion', unique_id=configured_slugify(rule['name']), device=ha_device_info)
    motion_settings = Settings(mqtt=mqtt_settings, entity=motion_sensor_info)
    motion_sensor = BinarySensor(motion_settings)

    parsed_rules[key] = {
        'parsed': parsed,
        'expression': rule['expression'],
        'ha_sensor': motion_sensor,
        'last_value': None
    }

@event_emitter.on('pose_result')
def lm_result(meta):
    for person in meta['persons']:
        # Check all rules against the person.
        for rule_name in parsed_rules:
            rule_data = parsed_rules[rule_name]
            evaluator.names = person
            result = evaluator.eval(rule_data['expression'], previously_parsed=rule_data['parsed'])

            def publish_result():
                if result:
                    print("Expression", rule_name, "(", rule_data['expression'], ")", "matched.")
                    rule_data['ha_sensor'].on()
                else:
                    rule_data['ha_sensor'].off()
                rule_data['last_value'] = result

            if rule_data['last_value'] is None:
                # First time it's been seen. Always publish it.
                publish_result()
            elif rule_data['last_value'] != result:
                publish_result()

