import logging
import threading
import time
import os
from dotenv import load_dotenv

import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def current_milli_time():
    return round(time.time() * 1000)

def landmarker_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('pose landmarker result: {}'.format(result))
    # print("Got result.")
    # print(result.pose_landmarks[0][16])
    age = current_milli_time() - timestamp_ms
    print("#####")
    print("Age", age)
    if len(result.pose_landmarks) == 1:
        vone = result.pose_landmarks[0]
        if len(vone) >= 16:
            # print("Right", vone[16].y, vone[12].y, angle_between_points(lmtp(vone[16]), lmtp(vone[12])))
            # print("Left", vone[15].y, vone[11].y, angle_between_points(lmtp(vone[15]), lmtp(vone[11])))

            # Ref: https://github.com/geaxgx/openvino_blazepose/blob/main/BlazeposeOpenvino.py#L61
            # Also https://github.com/geaxgx/openvino_blazepose/blob/main/BlazeposeOpenvino.py#L417

            right_arm_angle = angle_with_y((vone[14].x - vone[12].x, vone[14].y - vone[12].y))
            left_arm_angle = angle_with_y((vone[13].x - vone[11].x, vone[13].y - vone[11].y))
            right_pose = int((right_arm_angle +202.5) / 45) % 8
            left_pose = int((left_arm_angle +202.5) / 45) % 8
            letter = semaphore_flag.get((right_pose, left_pose), None)
            print("Right, left, pose, pose, letter", right_arm_angle, left_arm_angle, right_pose, left_pose, letter)


            # print("Left", vone[15].y, vone[11].y)

def thread_run_mediapipe(camera_url, modelfile):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=modelfile),
        running_mode=VisionRunningMode.VIDEO)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        print("Opening capture...")
        framecount = 0
        vcap = cv2.VideoCapture(camera_url)

        while True:
            # print("Capture...")
            ret, frame = vcap.read()
            if not ret:
                print("Error - failed to capture. Aborting.")
                break

            if framecount % 5 == 0:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                frame_timestamp_ms = current_milli_time()
                pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                landmarker_result(pose_landmarker_result, mp_image, frame_timestamp_ms)

            framecount += 1

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    load_dotenv()

    x = threading.Thread(target=thread_run_mediapipe, args=(os.getenv('CAMERA'), os.getenv('MODELFILE')))
    logging.info("Starting capture thread...")
    x.start()