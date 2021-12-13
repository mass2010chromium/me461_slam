import cv2
import numpy as np
import urllib.request
import json
import math
import time

import sys

do_move = False
if len(sys.argv) > 1:
    if sys.argv[1] == "--move":
        do_move = True

from utils.read_mat import SharedArray
from motion.purepursuit import PurePursuit
from motion.image_cspace import ImageCSpace, pos_to_map, robot_radius_px
import os
with open(os.path.expanduser("~/me461_slam/map_info.json"), "r") as map_info_file:
    map_info = json.load(map_info_file)

map_w = map_info['map_size']
map_center = map_w/2
map_scale = map_info['map_scale']
bound = (map_w / map_scale) / 2

cap = SharedArray('.slam.map', (map_w,map_w,3), np.uint8)

import math
from klampt.plan.cspace import CSpace, MotionPlan
from motionlib import vectorops as vo

MotionPlan.setOptions(type="fmm*")

def get_pose():
    position_info = urllib.request.urlopen("http://localhost:8080/pose_slam").read()
    position_json = json.loads(position_info)
    return (position_json['x'], position_json['y'], position_json['heading'])

kv = 3
def move_heading(target_angle):
    ok = 0
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
            ok += 1
            if ok > 30:
                break
        else:
            ok = 0
    
        data = json.dumps({'v': 0, 'w': np.clip(kv * angle_error, -1, 1)}).encode('utf-8')
        req = urllib.request.Request("http://localhost:8080/raw", data=data)
        resp = urllib.request.urlopen(req)

def move_to_pose(image, end_pose):
    position_info = urllib.request.urlopen("http://localhost:8080/pose_slam").read()
    position_json = json.loads(position_info)
    start_pose = (position_json['x'], position_json['y'], position_json['heading'])
    print("Recieved new target request", start_pose, end_pose, ", planning")
    if vo.norm(vo.sub(end_pose[:2], start_pose[:2])) < 0.1:
        print("Pure rotation")
        move_heading(target_info['heading'])
        return
    space = ImageCSpace(image, start_pose[:2])
    print(space.feasible(end_pose[:2]))
    planner = MotionPlan(space)
    try:
        planner.setEndpoints(start_pose[:2], end_pose[:2])
    except Exception as e:
        print(e)
        x, y = pos_to_map(end_pose)
        x = int(x)
        y = int(y)
        cv2.circle(image, (x, y), robot_radius_px, (0, 0, 255), 2)
        cv2.imshow("plan", image)
        cv2.waitKey(1)
        return
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
        return
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
        radius = 0.2
        speed = 0.2
        follower = PurePursuit(path, radius, speed, kv)
        cur = get_pose()
        print(cur)
        init_point, target_angle = follower.get_lookahead(cur)
        if init_point is not None:
            move_heading(target_angle)
        while True:
            ret, image = cap.read()
            cur = get_pose()
            space.update_map(image)
            if not space.feasible(cur):
                print("Encountered obstacle, replanning")
                move_to_pose(image, end_pose)
                return
            cv2.imshow("map", image)
            cv2.waitKey(50)
            done, cmd = follower.step(cur)
            if done:
                break
            data = json.dumps({'v': cmd[0], 'w': cmd[1]}).encode('utf-8')
            req = urllib.request.Request("http://localhost:8080/raw", data=data)
            resp = urllib.request.urlopen(req)

        move_heading(end_pose[2])

while True:
    ret, image = cap.read()
    cv2.imshow("map", image)
    cv2.waitKey(50)
    target_info = urllib.request.urlopen("http://localhost:8080/target").read()
    target_info = json.loads(target_info)
    if not target_info['new_request']:
        continue
    end_pose = (target_info['x'], target_info['y'], target_info['heading'])
    move_to_pose(image, end_pose)
