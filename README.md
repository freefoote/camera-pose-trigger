# Pose Trigger

A simple service to listen to a RTSP camera stream, and then trigger actions based on poses
seen inside the camera's view.

The original intent for this one is pure lazyness: being able to raise my arms to pause the TV playback,
because between four of us we can never seem to find the remote quickly. So lets make technology and machine
learning do this for us!

As usual, magic like this is based on the amazing work of others; my project is merely plumbing to hook a few
bits together.

Specifically, we use [Google's MediaPipe Pose Landmark Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
to do the heavy lifting, which also uses [OpenCV](https://opencv.org/). We also leverage OpenCV's ability to
read directly from RTSP streams.

## Setup

Create a virtualenv and install dependencies.

```bash
$ python3 -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```

Grab the model that matches what you had in mind. I used "Pose Landmarker (Full)" and got good results with it.

[MediaPipe Pose Landmark Detection Models](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models)

Copy over the environment file and adjust it (for running locally):

TODO: Complete documentation.


TODO:
* Frame sample count - make configurable
* Some way to load the expressions from file
* Connect to HA
* Auto download models
* Containerize