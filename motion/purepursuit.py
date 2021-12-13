import numpy as np
from motionlib import vectorops as vo
from math import pi, sqrt, atan2

def normalize_angle(angle):
    angle = angle % (2*pi)
    if angle < 0:
        angle += 2*pi
    return angle

def average_angle(a, b):
    a = normalize_angle(a)
    b = normalize_angle(b)
    min_angle = min(a, b)
    max_angle = max(a, b)
    err = max_angle - min_angle
    err2 = 2*pi - err
    if err2 < err:
        return max_angle + err2/2
    return min_angle + err/2

"""
Return how to turn from a to get to b.
So really its more like b-a
"""
def angle_diff_signed(a, b):
    a = normalize_angle(a)
    b = normalize_angle(b)
    if b > a:
        forward = b-a
        if forward < pi:
            return forward
        return forward - 2*pi
    if b < a:
        backward = a-b
        if backward < pi:
            return -backward
        return 2*pi-backward
    return 0

class PurePursuit:
    def __init__(self, path, radius, speed, kv):
        last = path[-1]
        prev = path[-2]
        direction = vo.unit(vo.sub(last, prev))
        path_pad_point = vo.add(last, vo.mul(direction, radius))
        self.__path = path + [path_pad_point]
        self.__radius = radius
        self.__speed = speed
        self.__kv = kv

    def get_lookahead(self, pose):
        center = pose[:2]
        heading = normalize_angle(pose[2])

        lookahead = None
        lookahead_angle = None
        for p1, p2 in zip(self.__path, self.__path[1:]):
            if p1 == p2:
                continue
            p1_c = vo.sub(p1, center)            
            p2_c = vo.sub(p2, center)            

            # Circle-line intersection calculation.
            segment = vo.sub(p2, p1)
            dx, dy = segment
            dsq = vo.normSquared(segment)
            D = vo.cross(p1_c, p2_c)

            discrim = self.__radius*self.__radius * dsq - D*D
            if discrim < 0:
                # No intersection.
                continue

            disc_root = sqrt(discrim)
            x1 = (D*dy + dx*disc_root) / dsq
            x2 = (D*dy - dx*disc_root) / dsq
            y1 = (-D*dx + dy*disc_root) / dsq
            y2 = (-D*dx - dy*disc_root) / dsq

            x_min = min(p1_c[0], p2_c[0])
            y_min = min(p1_c[1], p2_c[1])
            x_max = max(p1_c[0], p2_c[0])
            y_max = max(p1_c[1], p2_c[1])
            valid1 = ((x1 > x_min and x1 < x_max)
                    or (y1 > y_min and y1 < y_max))
            valid2 = ((x2 > x_min and x2 < x_max)
                    or (y2 > y_min and y2 < y_max))

            if valid1:
                lookahead = vo.add((x1, y1), center)
                lookahead_angle = atan2(dy, dx)
            if valid2:
                if (
                    lookahead is None
                    or (
                        vo.norm_L1(vo.sub((x1, y1), p2_c)) > vo.norm_L1(vo.sub((x2, y2), p2_c))
                    )
                ):
                    lookahead = vo.add((x2, y2), center)
                    lookahead_angle = atan2(dy, dx)
        return lookahead, lookahead_angle

    def step(self, pose):
        center = pose[:2]
        heading = normalize_angle(pose[2])
        lookahead, lookahead_angle = self.get_lookahead(pose)

        if lookahead is None or vo.norm(vo.sub(lookahead, self.__path[-1])) < self.__radius:
            return (True, None)

        dx, dy = vo.sub(lookahead, center)
        direct_angle = atan2(dy, dx)
        target_angle = average_angle(direct_angle, lookahead_angle)

        limit_error = angle_diff_signed(heading, lookahead_angle)
        angle_error = angle_diff_signed(heading, target_angle)
        max_angular_v = 1
        max_trans_v = 0.75
        angular_v = self.__kv * angle_error
        if abs(angular_v) > max_angular_v:
            angular_v = max_angular_v * angular_v / abs(angular_v)
        trans_v = self.__speed / (1)# + abs(limit_error) * self.__speed/2)
        if abs(trans_v) > max_trans_v:
            trans_v = max_trans_v * trans_v / abs(trans_v)

        return (False, (trans_v, angular_v))

if __name__ == "__main__":
    import urllib.request
    import json
    def get_pose():
        position_info = urllib.request.urlopen("http://localhost:8080/pose").read()
        position_json = json.loads(position_info)
        return (position_json['x'], position_json['y'], position_json['heading'])

    #path1 = [(0, 0), (0.5, 0), (0.5, 0.5), (0.25, 0.5)]
    #path2 = [(0.5, 0.5), (0, 0.5), (0, 0), (0.25, 0)]
    path1 = [(0, 0), (1, 0), (1, 1), (0.5, 1)]
    path2 = [(1, 1), (0, 1), (0, 0), (0.5, 0)]

    kv = 3
    radius = 0.2
    speed = 0.2
    for path in [path1, path2]:
        follower = PurePursuit(path, radius, speed, kv)
        while True:
            cur = get_pose()
            print(cur)
            done, cmd = follower.step(cur)
            if done:
                break
            data = json.dumps({'v': cmd[0], 'w': cmd[1]}).encode('utf-8')
            if abs(cmd[0]) > 1 or abs(cmd[1]) > 1:
                print(data)
            req = urllib.request.Request("http://localhost:8080/raw", data=data)
            resp = urllib.request.urlopen(req)
