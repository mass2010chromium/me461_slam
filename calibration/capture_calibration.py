import cv2
import time

print('connecting to camera')
cam = cv2.VideoCapture(0)

KEY_ESC = 27

img_counter = 0

path = "calibration_images/image{}.png"

while True:
    ret, image = cam.read()
    cv2.imshow('image', image)
    key = cv2.waitKey(1)
    if key == KEY_ESC:
        print("Esc")
        break
    if key == ord(' '):
        cv2.imwrite(path.format(img_counter), image)
        print(f"Image {img_counter}")
        img_counter += 1
