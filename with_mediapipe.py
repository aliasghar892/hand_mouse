import mediapipe as mp
import cv2
import pyautogui

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
model = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model),
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)
correct, frame = cam.read()
h = frame.shape[0]
w = frame.shape[1]
screen_size = pyautogui.size()

while (correct):
    correct, frame = cam.read()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = landmarker.detect(mp_image)
    if (len(result.hand_landmarks) > 0):
        for i in range(len(result.hand_landmarks[0])):
            if (i == 9):
                pyautogui.moveTo(screen_size[0]-int(result.hand_landmarks[0][i].x*screen_size[0]),
                                 int(result.hand_landmarks[0][i].y*screen_size[1]))
    #         cv2.putText(frame, str(i),
    #                     (int(result.hand_landmarks[0][i].x*w),
    #                      int(result.hand_landmarks[0][i].y*h)),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # cv2.imshow("cam", frame)
    # if (cv2.waitKey(1) & 0xFF == ord('q')):
    #     break
