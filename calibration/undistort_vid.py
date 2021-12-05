import cv2
import time
import json
import numpy as np

print('connecting to camera')
cam = cv2.VideoCapture(0)

with open("intrinsics.json") as _intrinsics:
    intrinsics = json.load(_intrinsics)

camera_mat = np.array(intrinsics["matrix"])
camera_dist = np.array(intrinsics["distortion"])

N = 20
while True:
    t0 = time.time()
    for i in range(N):
        ret, image = cam.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out = cv2.undistort(gray, camera_mat, camera_dist)
    t1 = time.time()
    print("FPS: ", N/(t1-t0), image.shape)
    #cv2.imshow('image', out)
    #cv2.waitKey(1)

