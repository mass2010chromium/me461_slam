"""
This file is the high level planner for the robot.
It does a DFS to explore the world while trying to move to a target position.
"""

import cv2
import numpy as np
import urllib.request
import json
import math
import time
import random
import requests
import sys
import os

from utils.read_mat import SharedArray
from motion.image_cspace import ImageCSpace, pos_to_map, robot_radius_px
from motion.simple_motions import get_pose, move_heading, spin_in_circle, set_velocity
from motion.purepursuit import PurePursuit
from motionlib import vectorops as vo

# Loading in some configuration.
with open("config.json", "r") as map_info_file:
    map_info = json.load(map_info_file)

map_w = map_info['map_size']
map_center = map_w/2
map_scale = map_info['map_scale']
bound = (map_w / map_scale) / 2
cap = SharedArray('.slam.map', (map_w,map_w,3), np.uint8)

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

    # First, spin in a circle.
    spin_in_circle()
    cur_real = vo.mul(current_pos, step_size)
    print(f"Expanding node: {current_pos}")

    # Next, find neighbors of the current node, and sort in ascending order according to distance from goal.
    targets = (tuple(vo.add(dir, current_pos)) for dir in direcs)
    targets_priority = sorted([(vo.norm(vo.sub(vo.mul(x, step_size), end_pos)), x) for x in targets])
    for _, target in targets_priority:
        # Skip visited targets.
        if target in visited:
            continue
        target_real = vo.mul(target, step_size)
        # Also skip targets that are outside our exploration radius.
        if max(np.abs(target_real)) > max_radius:
            continue
        print(f"Explore: {target}")

        # Construct a "path" (from current pose to the target pose in world coordinates),
        #     and try to move there.
        start_pos = get_pose()[:2]
        path = [start_pos, target_real]
        res = try_move_path(path)
        visited.add(target)
        goal_close = max(np.abs(vo.sub(end_pos, target_real))) < step_size/2
        if res:
            # Successfully moved to new cell. Check if we are close to the goal.
            if goal_close:
                # We are done, and we found the goal.
                return True, True

            # Recurse!
            done, found = dfs(target, end_pos, step_size, max_radius, visited)
            if done:
                return True, found

        if goal_close:
            # We are done. We tried moving to the goal but we could not.
            return True, False

        # Backtrack by moving from wherever we currently are, to our previous pose.
        print(f"backtrack {current_pos}")
        start_pos = get_pose()[:2]
        path = [start_pos, cur_real]
        try_move_path(path, False)
    return False, False


def try_move_path(path, col_check=True):
    """Try to move along a given path.
    Optionally, check for collisions and fail.
    Pause if you see a stop sign (temporary) or person (until person leaves).

    Parameters:
    --------------------
    path: List of waypoints to move to (world coords, meters, absolute)
    col_check: True to check for collisions and exit if colliding; false otherwise.
    """
    radius = 0.2
    speed = 0.1
    kv = 1.5
    # Create a path follower object (Pure pursuit algorithm).
    follower = PurePursuit(path, radius, speed, kv)

    # First point the robot in the correct direction.
    cur = get_pose()
    init_point, target_angle = follower.get_lookahead(cur)
    if init_point is not None:
        move_heading(target_angle)

    # Intialize collision checking data structure if needed.
    if col_check:
        ret, image = cap.read()
        space = ImageCSpace(image, cur[:2])
    
    # Follow the path!
    detectStopSign = 0
    while True:
        # First, check if we have detected any people/stop signs.
        personDetected = False
        cur = get_pose()
        returnResponse = requests.get("http://localhost:8080/detections").json()
        detectedObjNames = return
        detectedStopSign = False
        for responseObs in returnResponse:
            objectDetected = responseObs["name"]
            if objectDetected == "stop sign":
                detectedStopSign = True
            elif objectDetected == "person":
                personDetected = True

        # If detected stop sign: Wait 5s, then set flag to prevent repeat retriggering (within some time window)
        if detectedStopSign:
            if detectStopSign == 0:
                set_velocity(0, 0)
                print("stop sign")
                time.sleep(5)
            detectStopSign = 500
        elif detectStopSign > 0:
            detectStopSign -= 1

        # If detected person: Wait until person leaves frame
        if personDetected:
            set_velocity(0, 0)
            print("person")
            continue
        

        # Check for collisions (if enabled).
        if col_check:
            ret, image = cap.read()
            point, angle = follower.get_lookahead(cur)
            space.update_map(image)
            if not space.feasible(point):
                print("Encountered obstacle, failing")
                set_velocity(0, 0)

                # Show reason for collision for debugging.
                x, y = pos_to_map(cur)
                x = int(x)
                y = int(y)
                cv2.circle(image, (x, y), robot_radius_px, (0, 0, 255), 2)
                cv2.imshow("failure", image)
                cv2.waitKey(1)
                return False

        # Use path follower to compute a) if we are done and b) the velocity to use.
        done, cmd = follower.step(cur)
        if done:
            break
        set_velocity(*cmd)
        time.sleep(0.050)
    delta = vo.norm(vo.sub(cur[:2], path[-1][:2]))
    if delta > radius:
        return False
    return True

if __name__ == "__main__":
    # Do the actual DFS.
    visited = set()
    visited.add((0, 0))
    step_size = 0.5 # meters
    max_radius = 3
    dfs((0, 0), (1.5, 0), step_size, max_radius, visited)
