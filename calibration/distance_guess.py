import cv2
import time
import json
import numpy as np

with open("intrinsics.json") as _intrinsics:
    intrinsics = json.load(_intrinsics)

camera_mat = np.array(intrinsics["matrix"])
camera_dist = np.array(intrinsics["distortion"])

#test_im = cv2.imread("undistort.png")
test_im = cv2.imread("cpp_log/capture_000565.png")
test_im = cv2.undistort(test_im, camera_mat, camera_dist)
cv2.imwrite("undistort.png", test_im)
h, w, _ = test_im.shape

test_point_1 = (419, 382)
test_point_2 = (534, 385)
test_point_3 = (456, 355)
marked_im = cv2.circle(test_im, test_point_1, 25, (0, 0, 255), 2)
marked_im = cv2.circle(marked_im, test_point_2, 25, (0, 0, 255), 2)
marked_im = cv2.circle(marked_im, test_point_3, 25, (0, 0, 255), 2)

fx = camera_mat[0, 0]
fy = camera_mat[1, 1]
fudge_factor = 0.0820761712820892 / 0.075
fx *= fudge_factor
#fy /= fudge_factor
print(fx, fy)
y_real = -0.105 # 10.5cm above ground, the camera

def transform_point(cam_pt):
    x_cam, y_cam = cam_pt
    print(h/2)
    x_real = fy * y_real / (h/2 - y_cam)
    z_real = (x_cam - w/2) * x_real / fx
    return np.array((x_real, y_real, z_real))

transform_1 = transform_point(test_point_1)
transform_2 = transform_point(test_point_2)
transform_3 = transform_point(test_point_3)

print(transform_1)
print(transform_2)
print(transform_3)
print(np.linalg.norm(transform_1 - transform_2))

cv2.imshow("before", marked_im)
for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1000]:
    for z in [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]:
        x_im = int(w/2 - z * fx / x)
        y_im = int(h/2 - y_real * fy / x)
        if x_im >= 0 and x_im < w and y_im >= 0 and y_im < h:
            _marked_im = cv2.circle(marked_im, (x_im, y_im), 5, (255, 0, 0), 2)

cv2.imshow("marked", marked_im)
cv2.waitKey(0)
