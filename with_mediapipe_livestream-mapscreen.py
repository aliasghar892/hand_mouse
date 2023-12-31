import mediapipe as mp
import cv2
import mouse
import time
import math
import pyautogui


def euclideanDistance(a, b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
model = "models/hand_landmarker.task"

RESULT = None


def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global RESULT
    RESULT = result


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model),
    num_hands=1,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with HandLandmarker.create_from_options(options) as landmarker:
    cam = cv2.VideoCapture(0)
    correct, frame = cam.read()

    sensitivity = 1000
    screensize_x, screensize_y = pyautogui.size()

    eight_pos = (0, 0)  # first finger
    twelve_pos = (0, 0)  # second finger
    sixteen_pos = (0, 0)  # last finger
    four_pos = (0, 0)  # thumb

    while (correct):
        correct, frame = cam.read()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(round(time.time() * 1000))
        landmarker.detect_async(mp_image, timestamp)
        if (type(RESULT) != type(None) and len(RESULT.hand_landmarks) > 0):
            for i in range(len(RESULT.hand_landmarks[0])):
                if (i == 9):
                    mouse.move(
                        round(screensize_x-RESULT.hand_landmarks[0][i].x*screensize_x), round(RESULT.hand_landmarks[0][i].y*screensize_y), absolute=True)

                if (i == 4):
                    four_pos = (
                        RESULT.hand_landmarks[0][i].x, RESULT.hand_landmarks[0][i].y)
                if (i == 8):
                    eight_pos = (
                        RESULT.hand_landmarks[0][i].x, RESULT.hand_landmarks[0][i].y)
                if (i == 16):
                    sixteen_pos = (
                        RESULT.hand_landmarks[0][i].x, RESULT.hand_landmarks[0][i].y)
                if (i == 12):
                    twelve_pos = (
                        RESULT.hand_landmarks[0][i].x, RESULT.hand_landmarks[0][i].y)

            if (euclideanDistance(four_pos, eight_pos) < 0.1):
                if (mouse.is_pressed("left") == False):
                    mouse.press("left")
            else:
                mouse.release("left")

            if (euclideanDistance(twelve_pos, four_pos) < 0.1):
                mouse.right_click()

            if (euclideanDistance(sixteen_pos, four_pos) < 0.1):  # it will not work after some depth
                break

    cam.release()
