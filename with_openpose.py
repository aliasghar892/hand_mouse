import cv2
import pyautogui

cam = cv2.VideoCapture(0)

correct, img = cam.read()
h = int(img.shape[0]/2)
w = int(img.shape[1]/2)

proto = "pose_deploy.prototxt"
weights = "pose_iter_102000.caffemodel"
n_points = 22
net = cv2.dnn.readNetFromCaffe(proto, weights)

screen_size = pyautogui.size()

while (correct):
    correct, img = cam.read()
    imglow = cv2.resize(img, (w, h))
    blob = cv2.dnn.blobFromImage(
        imglow, 1/255, (w, h), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    for i in range(n_points):
        map = cv2.resize(out[0, i], (w*2, h*2))
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(map)
        if (maxval > 0.5):
            # if (i == 9):
            # pyautogui.moveTo(screen_size[0]-maxloc[0]*screen_size[0]/w,
            #                  maxloc[1]*screen_size[1]/h)
            cv2.putText(img, str(i), maxloc,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("cam", img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
