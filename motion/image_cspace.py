import cv2
import numpy as np
import math

from klampt.plan.cspace import CSpace

import json
import os
with open(os.path.expanduser("~/me461_slam/map_info.json"), "r") as map_info_file:
    map_info = json.load(map_info_file)

map_w = map_info['map_size']
map_center = map_w/2
map_scale = map_info['map_scale']
bound = (map_w / map_scale) / 2

robot_radius = 0.3
robot_radius_px = math.ceil(robot_radius * map_scale)
robot_ksize = 2*robot_radius_px+1
robot_kernel = np.zeros([robot_ksize, robot_ksize], dtype=np.uint8)
print(robot_kernel.shape)
cv2.circle(robot_kernel, (robot_radius_px, robot_radius_px), robot_radius_px, 1, -1)

def pos_to_map(pos):
    return (map_center + pos[0] * map_scale, map_center - pos[1] * map_scale)

class ImageCSpace(CSpace):
    def __init__(self, obstacle_map, start_pose):
        CSpace.__init__(self)
        self.bound = [(-bound, -bound), (bound, bound)]
        self.eps = 1e-3
        self.obstacle_map = obstacle_map[:, :, 0]
        x, y = pos_to_map(start_pose)
        x = int(x)
        y = int(y)
        self.start_point = (x, y)
        cv2.circle(obstacle_map, self.start_point, robot_radius_px+1, 0, -1)

    def update_map(self, map):
        self.obstacle_map = map[:, :, 0]
        cv2.circle(map, self.start_point, robot_radius_px+1, 0, -1)

    def feasible(self, q):
        px, py = pos_to_map(q)
        px = int(px)
        py = int(py)
        if (px < robot_radius_px or (px + robot_radius_px) >= map_w
            or py < robot_radius_px or (py + robot_radius_px) >= map_w
        ):
            print("wtf")
            return False
        image_section = self.obstacle_map[py-robot_radius_px:py+robot_radius_px+1,
                                          px-robot_radius_px:px+robot_radius_px+1] > 80
        collision_score = image_section.ravel().dot(robot_kernel.ravel())
        #print(q, collision_score)
        return bool(collision_score < 10)

