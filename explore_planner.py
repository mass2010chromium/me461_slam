import cv2
import numpy as np
import urllib.request
import json
import math
import time
import random

import sys

from utils.read_mat import SharedArray
from motion.image_cspace import ImageCSpace, pos_to_map, robot_radius_px
from motion.simple_motions import get_pose, move_heading, spin_in_circle
from motion.purepursuit import PurePursuit
from motionlib import vectorops as vo

with open("config.json", "r") as map_info_file:
    map_info = json.load(map_info_file)

map_w = map_info['map_size']
map_center = map_w/2
map_scale = map_info['map_scale']
bound = (map_w / map_scale) / 2
cap = SharedArray('.slam.map', (map_w,map_w,3), np.uint8)


# TODO: object detection (pause when person detected, stop for few sec and resume when stop sign detected)
data = json.dumps({'v': cmd[0], 'w': cmd[1]}).encode('utf-8')
req = urllib.request.Request("http://localhost:8080/raw", data=data)
resp = urllib.request.urlopen(req)

direcs = [[1, 0],[-1,0],[0,-1],[0,1]]

##### code for exploration and finding the target
def dfs(current_pos, end_pos, step_size, max_radius, visited):
    """Explore the world.
    
    Internal grid representation is an integer grid. Scaled by `step_size` to real world coords (meters).
    (This is to prevent floating point nonsense)
    
    Assumes we start at (0, 0), explores until it is at `end_pos` or it can't reach the end position, or
    it has exhausted its designated search space (a square with real "radius" (half side-length) `max_radius`).
    
    Parameters:
    --------------------
    current_pos: Where we are right now, grid coords.
    end_pos: Where we want to go, grid coords.
    step_size: Scaling from grid coords to real coords (meters).
    max_radius: Max real radius to explore.
    visited: Set of nodes that have been visited.
    """
    spin_in_circle()
    cur_real = vo.mul(current_pos, step_size)
    print(f"Expanding node: {current_pos}")
    targets = (tuple(vo.add(dir, current_pos)) for dir in direcs)
    targets_priority = sorted([(vo.norm(vo.sub(x, end_pos)), x) for x in targets])
    for _, target in targets_priority:
        if target in visited:
            continue
        target_real = vo.mul(target, step_size)
        if max(np.abs(target_real)) > max_radius:
            continue
        print(f"Explore: {target}")

        start_pos = get_pose()[:2]
        path = [start_pos, target_real]
        res = try_move_path(path)
        visited.add(target)
        if res:
            # Successfully moved to new cell.
            if target == end_pos:
                # We are done.
                return True, True
            done, found = dfs(target, end_pos, step_size, max_radius, visited)
            if done:
                return True, found

        if target == end_pos:
            # We are done.
            return True, False

        start_pos = get_pose()[:2]
        path = [start_pos, current_pos]
        try_move_path(path, False)
    return False, False


def try_move_path(path, col_check=True):
    radius = 0.2
    speed = 0.1
    kv = 1.5
    follower = PurePursuit(path, radius, speed, kv)
    cur = get_pose()
    init_point, target_angle = follower.get_lookahead(cur)
    if init_point is not None:
        move_heading(target_angle)
    if col_check:
        ret, image = cap.read()
        space = ImageCSpace(image, cur[:2])
    while True:
        cur = get_pose()

        if col_check:
            ret, image = cap.read()
            point, angle = follower.get_lookahead(cur)
            space.update_map(image)
            if not space.feasible(point):
                print("Encountered obstacle, failing")

                x, y = pos_to_map(cur)
                x = int(x)
                y = int(y)
                cv2.circle(image, (x, y), robot_radius_px, (0, 0, 255), 2)
                cv2.imshow("failure", image)
                cv2.waitKey(1)
                return False

        time.sleep(0.050)
        done, cmd = follower.step(cur)
        if done:
            break
        data = json.dumps({'v': cmd[0], 'w': cmd[1]}).encode('utf-8')
        req = urllib.request.Request("http://localhost:8080/raw", data=data)
        resp = urllib.request.urlopen(req)
    delta = vo.norm(vo.sub(cur[:2], path[-1][:2]))
    if delta > radius:
        return False
    return True

if __name__ == "__main__":
    visited = set()
    step_size = 0.5 # meters
    max_radius = 1
    dfs((0, 0), (1, 1), step_size, max_radius, visited)
