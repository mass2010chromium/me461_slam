import cv2
import math
import time
import json
import numpy as np
from scipy.signal import medfilt
from sklearn import cluster

from motionlib import so3, se3
from motionlib import vectorops as vo

import urllib.request

import sys
if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder = "./cpp_log/"

# lol hack
camera = folder == '--camera'

with open("intrinsics.json") as _intrinsics:
    intrinsics = json.load(_intrinsics)
fov_x = intrinsics['fovx']

if camera:
    print('connecting to camera')
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera_mat = np.array(intrinsics["matrix"])
camera_dist = np.array(intrinsics["distortion"])

#test_im = cv2.imread("undistort.png")
#test_im = cv2.imread("cpp_log/capture_000565.png")
#test_im = cv2.undistort(test_im, camera_mat, camera_dist)
#cv2.imwrite("undistort.png", test_im)

camera_pos = (0.04, 0.0325, 0.105)

h, w = (480, 640)

init_floor_px = (470, 320)

mid_x = camera_mat[0, 2]
mid_y = camera_mat[1, 2]

fx = camera_mat[0, 0]
fy = camera_mat[1, 1]
#fudge_factor = 0.0820761712820892 / 0.075
#fx /= fudge_factor
#fy /= fudge_factor
z_real = -0.105 # 10.5cm above ground, the camera
#z_real = -0.05

"""
Transform a point in pixel space into a point relative to the camera frame.
"""
def transform_point(cam_pt):
    x_cam, y_cam = cam_pt
    x_real = fy * z_real / (mid_y - y_cam)
    y_real = (mid_x - x_cam) * x_real / fx
    return (x_real, y_real, z_real)

def normalize_angle(angle):
    angle = angle % (2*math.pi)
    if angle < 0:
        angle += 2*math.pi
    return angle

"""
pre-normalize pls
"""
def angle_distance(a, b):
    max_angle = max(a, b)
    min_angle = min(a, b)
    err = max_angle - min_angle
    err2 = 2*math.pi - err
    return min(err, err2)

"""
Project a point in real space into the given camera pose.
"""
def project_point(cam_pose_inv, real_pt):
    x, y, z = se3.apply(cam_pose_inv, real_pt)
    x_im = int(mid_x - y * fx / x)
    y_im = int(mid_y - z_real * fy / x)
    return (x_im, y_im)

"""
Get a 4x4 camera pose from a robot pose.
For now we are assuming the encoders are perfect.
"""
def get_camera_pose(pose_obj):
    rotation = so3.from_axis_angle(([0, 0, 1], pose_obj["heading"]))
    return (rotation, vo.add((pose_obj["x"], pose_obj["y"], 0), so3.apply(rotation, camera_pos)))

"""
returns (segment distance, line distance)
"""
def point_to_segment(point, line):
    corner1 = line[:2]
    corner2 = line[2:]
    linevec = vo.sub(corner2, corner1)
    pointvec = vo.sub(point, corner1)
    area = vo.dot(linevec, pointvec)
    line_dist = abs(vo.cross(linevec, pointvec)) / vo.norm(linevec)
    if area < 0:
        # point is opposite corner1 w.r.t. corner2.
        return vo.norm(pointvec), line_dist
    pointvec2 = vo.sub(point, corner2)
    if vo.dot(linevec, pointvec2) > 0:
        # point is opposite corner2 w.r.t. corner1.
        return vo.norm(pointvec2), line_dist
    return line_dist, line_dist

def line_distance(l1, l2):
    data = np.array((
        point_to_segment(l1[:2], l2),
        point_to_segment(l1[2:], l2),
        point_to_segment(l2[:2], l1),
        point_to_segment(l2[2:], l1)
    ))
    angle1 = math.atan2(l1[3] - l1[1], l1[2] - l1[0])
    if angle1 < 2*math.pi:
        angle1 += 2*math.pi
    angle2 = math.atan2(l2[3] - l2[1], l2[2] - l2[0])
    if angle2 < 2*math.pi:
        angle2 += 2*math.pi
    max_angle = max(angle1, angle2)
    min_angle = min(angle1, angle2)
    angle_err = max_angle - min_angle
    angle_err2 = 2*math.pi - angle_err
    angle_err = min(angle_err, angle_err2)
    closest_dist = min(data[:, 0])
    return closest_dist * (1 + angle_err) + angle_err * (1 + closest_dist)

print(line_distance([352,16,330,13],[293,212, 311,212]))

def plot_lines(img, lines, color):
    for line in lines:
        cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color)

def merge_lines(lines):
    x_y_arr = np.array(lines)
    xvals = x_y_arr[:,0]
    yvals = x_y_arr[:,1]
    m, b = np.polyfit(xvals, x_y_arr[:,1], 1)
    min_x = np.min(xvals)
    max_x = np.max(xvals)
    min_y = np.min(yvals)
    max_y = np.max(yvals)
    if m > 0:
        return [min_x, min_y, max_x, max_y]
    return [min_x, max_y, max_x, min_y]
    min_y = m * min_x + b
    max_y = m * max_x + b
    return [min_x, min_y, max_x, max_y]

def dbscan_filter_lines(lines, existings, eps):
    dbscan = cluster.DBSCAN(eps, metric=line_distance, min_samples=1)
    all_lines = lines + [x[0] for x in existings]
    dbscan.fit(all_lines)
    y_pred = dbscan.labels_.astype(int)
    boxes = {}
    ret = []
    groupsizes = {}
    for group in y_pred:
        groupsizes[group] = groupsizes.get(group, 0) + 1
    for group, line, i in zip(y_pred, all_lines, range(len(all_lines))):
        if group == -1 or groupsizes[group] == 1:
            # "noise" but in this case we keep them lol?
            if i < len(lines):
                # Only if it wasn't old.
                ret.append([line, 1])
            else:
                existings[i-len(lines)][1] -= 1
                if existings[i-len(lines)][1] != 0:
                    # Some lifespan.
                    ret.append([line, existings[i-len(lines)][1]])
        else:
            if group in boxes:
                bbox = boxes[group]
                boxes[group][0] += [line[:2], line[2:]]
            else:
                boxes[group] = [[line[:2], line[2:]], False]
            if i >= len(lines):
                boxes[group][1] = True
    for k, v in boxes.items():
        lines, old = v
        if old:
            age = 8
        else:
            age = 1
        ret.append([merge_lines(lines), age])
    return ret

def split_image(grad_img, prev_transform_lines):
    h, w = grad_img.shape
    split_row = h//2
    yvals = np.empty(w)
    reversed = grad_img[::-1, :]
    _target = np.zeros((h - split_row, w), np.uint8)
    yvals = np.argmax(reversed, axis=0)
    for i in range(w):
        if yvals[i] < split_row:
            k = yvals[i]
            if reversed[k, i] != 0:
                _target[k, i] = 128

    prev_mask = np.zeros(grad_img.shape, dtype=np.uint8)
    plot_lines(prev_mask, prev_transform_lines, 1)
    prev_mask *= grad_img
    _target += prev_mask[w-1:split_row-1:-1, :]
    cv2.imshow("target", _target)
    target = _target#cv2.GaussianBlur(_target, (0, 0), 0.5) * 2
    match_lines = cv2.HoughLinesP(target, 5, 0.01, 50, minLineLength=10, maxLineGap=10)
    if match_lines is None:
        print("no lines found")
        return []
    print(len(match_lines), "lines found")
    lines = [l[0] for l in match_lines[:50]]

    _target[:, :] = 0
    plots = lines
    #plots = lines
    for line in plots:
        line[1] = h-1 - line[1]
        line[3] = h-1 - line[3]

    return plots

    offset = split_row // 8
    scores = np.zeros((split_row + 4, w))
    scores[2:-2, :] = grad_img[split_row:, :]
    pointers = np.zeros((split_row + 2, w), dtype=np.int32)
    scores[2:-2, :] *= np.array(range(offset, split_row + offset))[:, np.newaxis] / split_row
    #scores[1:-1, 0] += range(split_row)
    #scores[1:-1, -1] += range(split_row)
    #scores[1:-1, 0] *= 2
    #scores[1:-1, -1] *= 2
    for i in range(1, w):
        for j in range(1, split_row+1):
            save = 0
            save_idx = -1
            for k in range(j-2, j+3):
                if scores[k, i-1] > save:
                    save = scores[k, i-1]
                    save_idx = k
            scores[j, i] += save
            pointers[j, i] = save_idx

    max_row = np.argmax(scores[:, -1])
    grad_img[split_row + max_row-2, i] = 100000
    for i in range(w-2, -1, -1):
        max_row = pointers[max_row, i+1]
        grad_img[split_row + max_row-2, i] = 100000

def draw_robot(img, pose, pointer_scale=0.5):
    pose_center = pose[:2]
    pose_px = map_scale(pose_center)
    heading = pose[2]
    max_angle = heading + fov_x/2
    min_angle = heading - fov_x/2
    pose_px_x, pose_px_y = pose_px
    cv2.circle(img, pose_px, 5, (0, 255, 0), 1)
    cv2.line(img, pose_px,
                       map_scale((pose_x + pointer_scale * np.cos(heading),
                                  pose_y + pointer_scale * np.sin(heading))), (0, 255, 0))
    cv2.line(img, pose_px,
                       map_scale((pose_x + 100*np.cos(max_angle),
                                  pose_y + 100*np.sin(max_angle))), (255, 0, 0))
    cv2.line(img, pose_px,
                       map_scale((pose_x + 100*np.cos(min_angle),
                                  pose_y + 100*np.sin(min_angle))), (255, 0, 0))

plot_points = []
for x in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for y in [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]:
        x_im = int(mid_x - y * fx / x)
        y_im = int(mid_y - z_real * fy / x)
        if x_im >= 0 and x_im < w and y_im >= 0 and y_im < h:
            plot_points.append((x, y, z_real, x_im, y_im))

prev_pose = None
prev_points = None
prev_lines = []
prev_observe_mask = None

VIDEO_DELAY = 0
start_frame = 10
end_frame = 254
frame_count = start_frame

map_w = 400
map_center = map_w // 2
map_scaling = 25
map_img = np.ones((map_w, map_w, 3), dtype=np.uint8)* 255
cv2.circle(map_img, (map_center, map_center), int(0.5*map_scaling), (0, 0, 0), -1)
def map_scale(pt):
    return int(pt[0] * map_scaling + map_center), int(-pt[1] * map_scaling + map_center)

FEET_TO_METER = 0.3048

# Add this to heading to get "true heading".
estimated_rot_err = 0

saved_lines = []

keyframe_count = 0

pose = None
prev_head_tmp = None
pose_queue = []
while True:
    if camera:
        ret, _test_im = cam.read()
    else:
        _test_im = cv2.imread(folder + "/capture_{:06}.png".format(frame_count), cv2.IMREAD_COLOR)
        print("Frame", frame_count)
    test_im = _test_im#cv2.undistort(_test_im, camera_mat, camera_dist)
    if camera:
        new_pose = json.loads(urllib.request.urlopen("http://localhost:8080/pose").read())
    else:
        with open(folder + "/pose_{:06}.json".format(frame_count - VIDEO_DELAY)) as pose_file:
            new_pose = json.load(pose_file)
    new_pose['x'] *= FEET_TO_METER
    new_pose['y'] *= FEET_TO_METER
    if pose is None:
        pose = new_pose
        prev_head_tmp = pose['heading']
    else:
        pose_queue.append(new_pose)
        if len(pose_queue) == VIDEO_DELAY + 1:
            new_pose = pose_queue.pop(0)
        dx = new_pose['x'] - pose['x']
        dy = new_pose['y'] - pose['y']
        pose['x'] += dx * np.cos(estimated_rot_err) + dy * np.sin(estimated_rot_err)
        pose['y'] += dy * np.cos(estimated_rot_err) - dx * np.sin(estimated_rot_err)
        print("raw pose:", new_pose)
        #print("heading change:", new_pose['heading'] - prev_head_tmp)
        prev_head_tmp = new_pose['heading']
        pose['heading'] = new_pose['heading'] - estimated_rot_err
        pose['v'] = new_pose['v']
        pose['w'] = new_pose['w']

    marked_im = test_im.copy()

    #deriv = 50 * (np.abs(cv2.Laplacian(cv2.GaussianBlur(marked_im, (0, 0), 5), cv2.CV_64F, ksize=5)) > 30)
    #deriv = np.array(50 * (np.abs(cv2.Laplacian(cv2.GaussianBlur(marked_im, (0, 0), 3), cv2.CV_64F, ksize=5)) > 100), dtype=np.uint8)
    #deriv = np.array(50 * (np.abs(cv2.Laplacian(cv2.GaussianBlur(marked_im, (0, 0), 0.5), cv2.CV_64F, ksize=3)) > 100), dtype=np.uint8)
    deriv = cv2.Canny(marked_im, 125, 300) // 2
    
    cam_pose = get_camera_pose(pose)
    cam_inv = se3.inv(cam_pose)

    prev_transform_lines = []
    for line, _ in prev_lines:
        p1 = (*line[:2], 0)
        p2 = (*line[2:], 0)
        new_p1 = project_point(cam_inv, p1)
        new_p2 = project_point(cam_inv, p2)
        prev_transform_lines.append([*new_p1, *new_p2])
    lines = split_image(deriv, prev_transform_lines)

    observe_mask = np.zeros(map_img.shape[:2])
    disp_map = map_img.copy()
    new_lines = np.zeros(map_img.shape, dtype=np.uint8)

    plot_lines(deriv, lines, 255)
    #print(deriv.shape)
    tracked_points = []
    for x, y, z, x_im, y_im in plot_points:
        transformed_point = se3.apply(cam_pose, (x, y, z))
        tracked_points.append(transformed_point)
        marked_im = cv2.circle(marked_im, (x_im, y_im), 2, (255, 0, 0), 2)

    proj_lines = []
    for line in lines:
        p1 = line[:2]
        p2 = line[2:]
        new_p1 = se3.apply(cam_pose, transform_point(p1))
        new_p2 = se3.apply(cam_pose, transform_point(p2))
        proj_lines.append((*new_p1[:2], *new_p2[:2]))

    proj_lines = dbscan_filter_lines(proj_lines, prev_lines, 0.25)
    keyframe_count += 1
    if keyframe_count >= 5:
        saved_subset = []
        heading = normalize_angle(pose['heading'])
        pose_x = pose['x']
        pose_y = pose['y']
        for i, (line, score) in enumerate(saved_lines):
            x1, y1, x2, y2 = line
            v1 = vo.sub((x1, y1), pose_x)
            v2 = vo.sub((x2, y2), pose_y)
            angle1 = normalize_angle(math.atan2(v1[1], v1[0]))
            angle2 = normalize_angle(math.atan2(v2[1], v2[0]))
            if (angle_distance(angle1, heading) < fov_x
                or angle_distance(angle2, heading) < fov_x
            ):
                saved_subset.append(line)
            if (angle_distance(angle1, heading) < fov_x/2
                or angle_distance(angle2, heading) < fov_x/2
            ):
                pass
            else:
                saved_lines[i][1] += 1
        match_subset = []
        for line, score in proj_lines:
            if score > 2:
                match_subset.append(line)

        def loss(t, dx, dy):
            rotation = so3.from_axis_angle(([0, 0, 1], t))
            translation = (dx, dy, 0)
            transform = (rotation, translation)
            score = 0
            for move in match_subset:
                closest_line = None
                closest_dist = 0
                p1 = (*move[:2], 0)
                p2 = (*move[2:], 0)
                p1_t = se3.apply(transform, p1)[:2]
                p2_t = se3.apply(transform, p2)[:2]
                move_trans = p1_t + p2_t    # concat
                for match in saved_subset:
                    dist = line_distance(move_trans, match)
                    if closest_line is None or dist < closest_dist:
                        closest_line = match
                        closest_dist = dist

                if closest_dist > 0.3:
                    closest_dist = 0.3
                score += closest_dist
            return score + vo.norm(transform[1]) + abs(t/10)

        best_tup = (0,0,0)
        best_loss = loss(*best_tup)

        theta_spread = 0.3
        delta_spread = 0.05
        for j in range(4):
            thetas = np.linspace(best_tup[0] - theta_spread, best_tup[0] + theta_spread, 5)
            delta_x = np.linspace(best_tup[1] - delta_spread, best_tup[1] + delta_spread, 5)
            delta_y = np.linspace(best_tup[2] - delta_spread, best_tup[2] + delta_spread, 5)
            prev_loss = best_loss
            for t in thetas:
                for dx in delta_x:
                    for dy in delta_y:
                        calc_loss = loss(t, dx, dy)
                        if calc_loss < best_loss:
                            best_tup = (t, dx, dy)
                            best_loss = calc_loss
            theta_spread /= 2
            delta_spread /= 2
            print("refining...", best_tup, best_loss)
        print("Computed transform:", best_tup)

        rotation = so3.from_axis_angle(([0, 0, 1], best_tup[0]))
        translation = (*best_tup[1:], 0)
        best_transform = (rotation, translation)
        transformed_lines = []
        for move in match_subset:
            p1 = (*move[:2], 0)
            p2 = (*move[2:], 0)
            p1_t = se3.apply(best_transform, p1)[:2]
            p2_t = se3.apply(best_transform, p2)[:2]
            transformed_lines.append(p1_t + p2_t)
        
        saved_lines = dbscan_filter_lines(transformed_lines, saved_lines, 0.25)
        for i in range(len(saved_lines)):
            if saved_lines[i][1] > 2:
                saved_lines[i][1] = 2
        print(len(saved_lines))
        
        tmp_map = disp_map.copy()
        pose['x'] += best_tup[1]
        pose['y'] += best_tup[2]
        pose['heading'] += best_tup[0]
        estimated_rot_err -= best_tup[0]
        plot_lines(tmp_map, [map_scale(l[0][:2]) + map_scale(l[0][2:]) for l in saved_lines], (0, 255, 0))
        draw_robot(tmp_map, (pose['x'], pose['y'], heading))
        cv2.imshow("match", tmp_map)

        proj_lines = []
        for i, (line, score) in enumerate(saved_lines):
            x1, y1, x2, y2 = line
            v1 = vo.sub((x1, y1), pose_x)
            v2 = vo.sub((x2, y2), pose_y)
            angle1 = normalize_angle(math.atan2(v1[1], v1[0]))
            angle2 = normalize_angle(math.atan2(v2[1], v2[0]))
            if (angle_distance(angle1, heading) < fov_x
                or angle_distance(angle2, heading) < fov_x
            ):
                proj_lines.append([line, 2])
        keyframe_count = 0

    print(pose)
    scaled_lines = []
    for l, _ in proj_lines:
        p1 = map_scale(l[:2])
        p2 = map_scale(l[2:])
        scaled_lines.append((*p1, *p2))

    pose_x = pose['x']
    pose_y = pose['y']
    pose_center = (pose_x, pose_y)
    pose_px = map_scale(pose_center)
    pose_px_x, pose_px_y = pose_px
    zero_mask = np.zeros(map_img.shape[:2])
    max_depth = 1
    scale = map_scaling * max_depth
    circle_mask = zero_mask.copy()
    cv2.circle(circle_mask, pose_px, scale*3, 0.25, -1)
    cv2.circle(circle_mask, pose_px, scale*2, 0.5, -1)
    cv2.circle(circle_mask, pose_px, scale, 1, -1)
    cv2.circle(circle_mask, pose_px, int(0.28 * map_scaling), 0, -1)

    points_angles = []
    heading = normalize_angle(pose['heading'])
    max_angle = heading + fov_x/2
    min_angle = heading - fov_x/2
    unit_heading = (np.cos(heading), -np.sin(heading))
    # mmmmm rotation reversed due to left handed coord sys
    heading_perp = (np.sin(heading), np.cos(heading))
    hold_id = 1
    for x1, y1, x2, y2 in scaled_lines:
        v1 = vo.sub((x1, y1), pose_px)
        v2 = vo.sub((x2, y2), pose_px)
        v1_l = vo.norm(v1)
        v2_l = vo.norm(v2)
        angle1 = np.arccos(vo.dot(heading_perp, v1) / v1_l) - np.pi/2
        angle2 = np.arccos(vo.dot(heading_perp, v2) / v2_l) - np.pi/2
        append_points = []
        #if abs(angle1) < fov_x / 2 and vo.dot(unit_heading, v1) > 0:
        if vo.dot(unit_heading, v1) > 0:
            append_points.append([angle1, v1_l, 0])
        #if abs(angle2) < fov_x / 2 and vo.dot(unit_heading, v2) > 0:
        if vo.dot(unit_heading, v2) > 0:
            append_points.append([angle2, v2_l, 0])
        append_points.sort()
        if len(append_points) == 2:
            append_points[0][-1] = -hold_id
            append_points[1][-1] = hold_id
            hold_id += 1
        points_angles += append_points
    if len(points_angles) > 0:
        points_angles.sort()
        r_start = 0
        for angle, d, _ in points_angles:
            if angle > -fov_x/2:
                r_start = d
                break
        r_end = 0
        for angle, d, _ in points_angles[::-1]:
            if angle < fov_x/2:
                r_end = d
                break
        # More rotation inversion
        filter_points = [[-fov_x/2, r_start, 0]] + points_angles + [[fov_x/2, r_end, 0]]
        active_set = {}
        out_points = []
        for angle, d, hold in filter_points:
            _d = d
            for k, v in active_set.items():
                if v < d:
                    d = v
            angle += heading
            out_points.append([int(pose_px_x + math.cos(angle) * d),
                               int(pose_px_y - math.sin(angle) * d)])
            if hold < 0:
                active_set[hold] = d
            elif hold > 0:
                del active_set[-hold]
        cv2.fillPoly(observe_mask, pts=[np.array(out_points + [pose_px], dtype=np.int32)], color=1)
        triangle = np.array([(0, 0),
                    vo.mul((math.cos(max_angle), -math.sin(max_angle)), 100),
                    vo.mul((math.cos(min_angle), -math.sin(min_angle)), 100)], dtype=np.int32) + pose_px
        angle_mask = zero_mask.copy()
        cv2.fillPoly(angle_mask, pts=[triangle], color=1)
        observe_mask *= circle_mask * angle_mask
    cv2.imshow("mask", observe_mask)
    plot_lines(disp_map, scaled_lines, (0, 0, 255))
    plot_lines(new_lines, scaled_lines, (255, 255, 255))

    if prev_points is not None:
        for point in prev_points:
            x_im, y_im = project_point(cam_inv, point)
            if x_im >= 0 and x_im < w and y_im >= 0 and y_im < h:
                marked_im = cv2.circle(marked_im, (x_im, y_im), 5, (0, 0, 255), 2)
    draw_robot(disp_map, (pose_x, pose_y, heading))
    cv2.imshow("map", disp_map)
    cv2.imshow("before", deriv / 200)
    cv2.imshow("marked", marked_im)
    if camera:
        key = cv2.waitKeyEx(1)
    else:
        key = cv2.waitKeyEx(0)
    if key == 65361:
        # left arrow
        if frame_count > 0:
            frame_count -= 1
    elif key == 27 or key == ord('q'):
        break
    else:
        frame_count += 1
    map_img -= np.array(map_img * observe_mask[:, :, np.newaxis] * 0.15, dtype=np.uint8)
    map_img += np.array(new_lines * observe_mask[:, :, np.newaxis], dtype=np.uint8)

    prev_pose = cam_pose
    prev_points = tracked_points
    prev_lines = proj_lines
    prev_observe_mask = observe_mask
