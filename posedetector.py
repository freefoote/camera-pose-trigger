import logging
import threading
import time
import os
import math

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

def current_milli_time():
    return round(time.time() * 1000)

# Borrowed from https://github.com/geaxgx/openvino_blazepose/blob/main/BlazeposeOpenvino.py#L419
# with thanks!
def angle_with_y(v):
    # v: 2d vector (x,y)
    # Returns angle in degree ofv with y-axis of image plane
    if v[1] == 0:
        return 90
    angle = math.atan2(v[0], v[1])
    return math.degrees(angle)

# Borrowed from https://github.com/geaxgx/openvino_blazepose/blob/main/BlazeposeOpenvino.py#L60
# with thanks!
semaphore_flag = {
    (3,4):'A', (2,4):'B', (1,4):'C', (0,4):'D',
    (4,7):'E', (4,6):'F', (4,5):'G', (2,3):'H',
    (0,3):'I', (0,6):'J', (3,0):'K', (3,7):'L',
    (3,6):'M', (3,5):'N', (2,1):'O', (2,0):'P',
    (2,7):'Q', (2,6):'R', (2,5):'S', (1,0):'T',
    (1,7):'U', (0,5):'V', (7,6):'W', (7,5):'X',
    (1,6):'Y', (5,6):'Z'
}

# Ref: https://github.com/geaxgx/openvino_blazepose/blob/main/BlazeposeOpenvino.py#L61
# Also https://github.com/geaxgx/openvino_blazepose/blob/main/BlazeposeOpenvino.py#L417
def calculate_arm_angle(point1, point2):
    return math.floor(angle_with_y((point1.x - point2.x, point1.y - point2.y)))

def calculate_octant(angle):
    return int((angle +202.5) / 45) % 8

def extract_angles_from_poses(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int, event_emitter):
    age = current_milli_time() - timestamp_ms
    emit_data = {}
    emit_data['seen_people'] = len(result.pose_landmarks)
    emit_data['time_to_infer_ms'] = age

    persons = []

    for person in result.pose_landmarks:
        if len(person) >= 16:
            # What numbers are what points from the pose? From here:
            # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
            right_arm_angle = math.floor(angle_with_y((person[14].x - person[12].x, person[14].y - person[12].y)))
            left_arm_angle = math.floor(angle_with_y((person[13].x - person[11].x, person[13].y - person[11].y)))
            right_pose = int((right_arm_angle +202.5) / 45) % 8
            left_pose = int((left_arm_angle +202.5) / 45) % 8
            letter = semaphore_flag.get((right_pose, left_pose), None)

            person_emit_data = {}
            person_emit_data['right_arm_upper_angle'] = calculate_arm_angle(person[14], person[12])
            person_emit_data['left_arm_upper_angle'] = calculate_arm_angle(person[13], person[11])
            person_emit_data['right_arm_upper_octant'] = calculate_octant(person_emit_data['right_arm_upper_angle'])
            person_emit_data['left_arm_upper_octant'] = calculate_octant(person_emit_data['left_arm_upper_angle'])

            person_emit_data['right_arm_whole_angle'] = calculate_arm_angle(person[16], person[12])
            person_emit_data['left_arm_whole_angle'] = calculate_arm_angle(person[15], person[11])
            person_emit_data['right_arm_whole_octant'] = calculate_octant(person_emit_data['right_arm_whole_angle'])
            person_emit_data['left_arm_whole_octant'] = calculate_octant(person_emit_data['left_arm_whole_angle'])

            person_emit_data['semaphore_letter'] = letter

            persons.append(person_emit_data)

    emit_data['persons'] = persons

    event_emitter.emit('pose_result', emit_data)

# Ref: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image

def thread_run_pose_detector(event_emitter, camera_url, modelfile):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=modelfile),
        running_mode=VisionRunningMode.VIDEO)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        logging.info("Opening capture...")
        framecount = 0
        vcap = cv2.VideoCapture(camera_url)
        logging.info("Capture has been opened.")
        keep_capturing = True

        @event_emitter.on('pose_kill_thread')
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
                pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                extract_angles_from_poses(pose_landmarker_result, mp_image, frame_timestamp_ms, event_emitter)
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)

                event_emitter.emit('raw_frame', mp_image)
                event_emitter.emit('pose_annotated_frame', annotated_image)

            framecount += 1

        vcap.release()

if __name__ == "__main__":
    from dotenv import load_dotenv
    from pyee.base import EventEmitter

    load_dotenv()

    ee = EventEmitter()

    @ee.on('pose_result')
    def lm_result_simple(foo):
        print(foo)

    @ee.on('pose_annotated_frame')
    def lm_ann_frame(frame):
        cv2.imwrite('out.jpg', frame)

    x = threading.Thread(target=thread_run_pose_detector, args=(ee, os.getenv('CAMERA'), os.getenv('MODELFILE')))
    logging.info("Starting capture thread...")
    x.start()