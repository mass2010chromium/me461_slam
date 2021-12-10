import cv2
import numpy as np

from utils.read_mat import SharedArray

cap = SharedArray('.slam.map', (400,400,3), np.uint8)

while True:
    ret, image = cap.read()
    cv2.imshow("map", image)
    cv2.waitKey(1)
