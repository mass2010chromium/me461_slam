import cv2
import numpy as np
import urllib.request
import json
import math
import time
import random

import sys

from utils.read_mat import SharedArray
from motion.image_cspace import ImageCSpace
from motionlib import vectorops as vo

with open("map_info.json", "r") as map_info_file:
    map_info = json.load(map_info_file)


map_w = map_info['map_size']
map_center = map_w/2
map_scale = map_info['map_scale']
bound = (map_w / map_scale) / 2
cap = SharedArray('.slam.map', (map_w,map_w,3), np.uint8)

def get_planner_status():
    planner_status = urllib.request.urlopen("http://localhost:8080/planner_status").read()
    planner_status = json.loads(planner_status)
    return planner_status['status']

def send_target(x, y, t):
    data = json.dumps({'x': x, 'y': y, 't': t}).encode('utf-8')
    req = urllib.request.Request("http://localhost:8080/target", data=data)
    resp = urllib.request.urlopen(req)
    status = get_planner_status()
    for i in range(5):
        if status == 2:
            return False
        if status == 1:
            break
        time.sleep(1)
        status = get_planner_status()
    print("Planner recieved command")
    while status == 1:
        print("Robot moving...")
        status = get_planner_status()
        time.sleep(1)
    print("move finished, status=", status)
    return status == 0

moved_distance = 100
while True:
    print("Requesting planner status:")
    status = get_planner_status()
    if status == 1:
        print("Robot is moving...")
        time.sleep(5)
        continue
    if status == 2:
        print("Planner failed on previous iteration, continuing anyway")

    position_info = urllib.request.urlopen("http://localhost:8080/pose_slam").read()
    position_json = json.loads(position_info)
    start_pose = (position_json['x'], position_json['y'], position_json['heading'])

    ret, image = cap.read()
    space = ImageCSpace(image, start_pose[:2])
    min_distance = 0.5
    for i in range(100):
        angle = random.random() * 2*math.pi
        radius = random.random()/2 + min_distance
        delta = (radius*math.cos(angle), radius*math.sin(angle))
        target_pos = vo.add(delta, start_pose[:2])
        if not space.feasible(target_pos):
            min_distance -= 0.01
            continue
        print(target_pos)
        status = send_target(target_pos[0], target_pos[1], angle)
        if not status:
            break
        moved_distance += radius
        if moved_distance >= 0.5:
            send_target(999, 0, 0)
            moved_distance = 0
        break

