import urllib.request
import json
import math
import numpy as np

def get_pose():
    """
    Helper function for getting pose info from the server.
    Makes an http request and returns the result in a tuple.
    
    Return:
        (x, y, theta) global pose.
    """
    position_info = urllib.request.urlopen("http://localhost:8080/pose_slam").read()
    position_json = json.loads(position_info)
    return (position_json['x'], position_json['y'], position_json['heading'])

kv = 1.5
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
                data = json.dumps({'v': 0, 'w': 0}).encode('utf-8')
                req = urllib.request.Request("http://localhost:8080/raw", data=data)
                resp = urllib.request.urlopen(req)
                break
        else:
            ok = 0

        data = json.dumps({'v': 0, 'w': np.clip(kv * angle_error, -1, 1)}).encode('utf-8')
        req = urllib.request.Request("http://localhost:8080/raw", data=data)
        resp = urllib.request.urlopen(req)

def spin_in_circle():
    cur = get_pose()
    heading = cur[2]
    target_angle = heading + 2*math.pi
    while True:
        cur = get_pose()
        heading = cur[2]
        angle_error = target_angle - heading
    
        if abs(angle_error) < 0.1:
            ok += 1
            if ok > 30:
                data = json.dumps({'v': 0, 'w': 0}).encode('utf-8')
                req = urllib.request.Request("http://localhost:8080/raw", data=data)
                resp = urllib.request.urlopen(req)
                break
        else:
            ok = 0

        data = json.dumps({'v': 0, 'w': np.clip(kv * angle_error, -1, 1)}).encode('utf-8')
        req = urllib.request.Request("http://localhost:8080/raw", data=data)
        resp = urllib.request.urlopen(req)
