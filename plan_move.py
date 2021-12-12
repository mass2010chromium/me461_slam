import cv2
import numpy as np
import urllib.request
import json
import math

import sys

do_move = False
if len(sys.argv) > 1:
    if sys.argv[1] == "--move":
        do_move = True

from utils.read_mat import SharedArray
from motion.purepursuit import PurePursuit

with open("map_info.json", "r") as map_info_file:
    map_info = json.load(map_info_file)

map_w = map_info['map_size']
map_center = map_w/2
map_scale = map_info['map_scale']
bound = (map_w / map_scale) / 2
cap = SharedArray('.slam.map', (map_w,map_w,3), np.uint8)

robot_radius = 0.2
robot_radius_px = math.ceil(robot_radius * map_scale)
robot_ksize = 2*robot_radius_px+1
robot_kernel = np.zeros([robot_ksize, robot_ksize], dtype=np.uint8)
print(robot_kernel.shape)
cv2.circle(robot_kernel, (robot_radius_px, robot_radius_px), robot_radius_px, 1, -1)

def pos_to_map(pos):
    return (map_center + pos[0] * map_scale, map_center - pos[1] * map_scale)

import math
from klampt.plan.cspace import CSpace, MotionPlan
from motionlib import vectorops as vo

class ImageCSpace(CSpace):
    def __init__(self, obstacle_map, start_pose):
        CSpace.__init__(self)
        self.bound = [(-bound, -bound), (bound, bound)]
        self.eps = 1e-3
        self.obstacle_map = obstacle_map[:, :, 0]
        x, y = pos_to_map(start_pose)
        x = int(x)
        y = int(y)
        cv2.circle(obstacle_map, (x, y), robot_radius_px+1, 0, -1)

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
                                          px-robot_radius_px:px+robot_radius_px+1] > 150
        collision_score = image_section.ravel().dot(robot_kernel.ravel())
        #print(q, collision_score)
        return bool(collision_score < 40)

MotionPlan.setOptions(type="fmm*")

def get_pose():
    position_info = urllib.request.urlopen("http://localhost:8080/pose").read()
    position_json = json.loads(position_info)
    return (position_json['x'], position_json['y'], position_json['heading'])

kv = 3
def move_heading(target_angle):
    while True:
        cur = get_pose()
        heading = cur[2] % (2*math.pi)
        angle_error = target_angle - heading
        if angle_error > 2*math.pi:
            angle_error -= 2*math.pi
        elif angle_error < -2*math.pi:
            angle_error += 2*math.pi
    
        if angle_error > math.pi:
            angle_error = angle_error - 2*math.pi
        if angle_error < -math.pi:
            angle_error = 2*math.pi + angle_error
    
        if abs(angle_error) < 0.1:
            break
    
        data = json.dumps({'v': 0, 'w': np.clip(kv * angle_error, -1, 1)}).encode('utf-8')
        req = urllib.request.Request("http://localhost:8080/raw", data=data)
        resp = urllib.request.urlopen(req)

while True:
    ret, image = cap.read()
    cv2.imshow("map", image)
    cv2.waitKey(1)
    target_info = urllib.request.urlopen("http://localhost:8080/target").read()
    target_info = json.loads(target_info)
    if not target_info['new_request']:
        continue
    end_pose = (target_info['x'], target_info['y'])
    position_info = urllib.request.urlopen("http://localhost:8080/pose").read()
    position_json = json.loads(position_info)
    start_pose = (position_json['x'], position_json['y'])
    print("Recieved new target request", start_pose, end_pose, ", planning")
    if vo.norm(vo.sub(end_pose, start_pose)) < 0.1:
        print("Pure rotation")
        move_heading(target_info['heading'])
        continue
    space = ImageCSpace(image, start_pose)
    print(space.feasible(end_pose))
    planner = MotionPlan(space)
    try:
        planner.setEndpoints(start_pose, end_pose)
    except Exception as e:
        print(e)
        x, y = pos_to_map(end_pose)
        x = int(x)
        y = int(y)
        cv2.circle(image, (x, y), robot_radius_px, (0, 0, 255), 2)
        cv2.imshow("plan", image)
        cv2.waitKey(1)
        continue
    path = []
    G = None
    print("Planning...", end="", flush=True)
    for i in range(3):
        planner.planMore(1)
        path = planner.getPath()
        G = planner.getRoadmap()
        print(".", end="", flush=True)
    if path is None or len(path) == 0:
        print("Planner failed.")
        continue
    print("!")
    print("Path has", len(path), "waypoints.")
    for q in path:
        px, py = pos_to_map(q)
        px = int(px)
        py = int(py)
        cv2.circle(image, (px, py), 1, (0, 0, 255), 1)
    cv2.imshow("plan", image)
    cv2.waitKey(1)
    if do_move:
        radius = 0.1
        speed = 0.1
        follower = PurePursuit(path, radius, speed, kv)
        while True:
            cur = get_pose()
            done, cmd = follower.step(cur)
            if done:
                break
            data = json.dumps({'v': cmd[0], 'w': cmd[1]}).encode('utf-8')
            req = urllib.request.Request("http://localhost:8080/raw", data=data)
            resp = urllib.request.urlopen(req)

        target_angle = target_info['heading']
        move_heading(target_angle)
