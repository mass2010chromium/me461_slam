import cv2
import time
import json
import numpy as np
from scipy.signal import medfilt
from sklearn import cluster

from motionlib import so3, se3
from motionlib import vectorops as vo

with open("intrinsics.json") as _intrinsics:
    intrinsics = json.load(_intrinsics)
fov_x = intrinsics['fovx']

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
def point_to_line(point, line):
    corner1 = line[:2]
    corner2 = line[2:]
    linevec = vo.sub(corner2, corner1)
    pointvec = vo.sub(point, corner1)
    area = vo.dot(linevec, pointvec)
    if area < 0:
        # point is opposite corner1 w.r.t. corner2.
        return vo.norm(pointvec), -area / vo.norm(linevec)
    pointvec2 = vo.sub(point, corner2)
    if vo.dot(linevec, pointvec2) > 0:
        # point is opposite corner2 w.r.t. corner1.
        return vo.norm(pointvec2), area / vo.norm(linevec)
    dist = area / vo.norm(linevec)
    return dist, dist
def line_distance(l1, l2):
    data = np.array((
        point_to_line(l1[:2], l2),
        point_to_line(l1[2:], l2),
        point_to_line(l2[:2], l1),
        point_to_line(l2[2:], l1)
    ))
    return min(data[:, 0])*5 + max(data[:, 1])

dbscan = cluster.DBSCAN(eps=125, metric=line_distance, min_samples=1)

def plot_lines(img, lines, color):
    for line in lines:
        cv2.line(img, line[:2], line[2:], color)

def split_image(grad_img):
    h, w = grad_img.shape
    split_row = h//2
    yvals = np.empty(w)
    reversed = grad_img[::-1, :]
    _target = np.zeros((h - split_row, w), np.uint8)
    yvals = np.argmax(reversed, axis=0)
    for i in range(w):
        if yvals[i] < split_row:
            k = yvals[i]
            while reversed[k, i] != 0:
                _target[k, i] = 128
                k += 1
                if k == split_row:
                    break
    target = _target#cv2.GaussianBlur(_target, (0, 0), 0.5) * 2
    lines = [l[0] for l in cv2.HoughLinesP(target, 5, 0.01, 50, minLineLength=10, maxLineGap=20)[:100]]

    _target[:, :] = 0
    dbscan.fit(lines)
    y_pred = dbscan.labels_.astype(int)
    boxes = {}
    plots = []
    for group, line in zip(y_pred, lines):
        if group == -1:
            plots.append(line)
        else:
            if group in boxes:
                bbox = boxes[group]
                boxes[group] += [line[:2], line[2:]]
            else:
                boxes[group] = [line[:2], line[2:]]
    for k, v in boxes.items():
        x_y_arr = np.array(v)
        xvals = x_y_arr[:,0]
        m, b = np.polyfit(xvals, x_y_arr[:,1], 1)
        min_x = np.min(xvals)
        max_x = np.max(xvals)
        min_y = int(m * min_x + b)
        max_y = int(m * max_x + b)
        if min_y >= 0 and max_y < (h - split_row):
            plots.append([min_x, min_y, max_x, max_y])
    #print(y_pred)
    #print(len(lines), len(plots))
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

plot_points = []
for x in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for y in [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]:
        x_im = int(mid_x - y * fx / x)
        y_im = int(mid_y - z_real * fy / x)
        if x_im >= 0 and x_im < w and y_im >= 0 and y_im < h:
            plot_points.append((x, y, z_real, x_im, y_im))

prev_pose = None
prev_points = None
prev_lines = None
prev_observe_mask = None

VIDEO_DELAY = 4
start_frame = 53
end_frame = 254
i = start_frame

map_w = 400
map_img = np.zeros((map_w, map_w, 3), dtype=np.uint8)
map_center = map_w // 2
map_scaling = 25
def map_scale(pt):
    return int(pt[0] * map_scaling + map_center), int(-pt[1] * map_scaling + map_center)

FEET_TO_METER = 0.3048

# Add this to heading to get "true heading".
estimated_rot_err = 0

pose = None
prev_head_tmp = None
while True:
    _test_im = cv2.imread("cpp_log/capture_{:06}.png".format(i), cv2.IMREAD_GRAYSCALE)
    test_im = cv2.undistort(_test_im, camera_mat, camera_dist)
    with open("cpp_log/pose_{:06}.json".format(i - VIDEO_DELAY)) as pose_file:
        if pose is None:
            pose = json.load(pose_file)
            prev_head_tmp = pose['heading']
        else:
            new_pose = json.load(pose_file)
            dx = new_pose['x'] - pose['x']
            dy = new_pose['y'] - pose['y']
            pose['x'] += dx * np.cos(estimated_rot_err) + dy * np.sin(estimated_rot_err)
            pose['y'] += dy * np.cos(estimated_rot_err) - dx * np.sin(estimated_rot_err)
            print("raw pose:", new_pose)
            print("heading change:", new_pose['heading'] - prev_head_tmp)
            prev_head_tmp = new_pose['heading']
            pose['heading'] = new_pose['heading'] - estimated_rot_err
            pose['v'] = new_pose['v']
            pose['w'] = new_pose['w']
    pose['x'] *= FEET_TO_METER
    pose['y'] *= FEET_TO_METER

    marked_im = test_im.copy()

    #deriv = 50 * (np.abs(cv2.Laplacian(cv2.GaussianBlur(marked_im, (0, 0), 5), cv2.CV_64F, ksize=5)) > 30)
    deriv = np.array(50 * (np.abs(cv2.Laplacian(cv2.GaussianBlur(marked_im, (0, 0), 3), cv2.CV_64F, ksize=5)) > 100), dtype=np.uint8)
    lines = split_image(deriv)
    plot_lines(deriv, lines, 255)
    print(deriv.shape)
    tracked_points = []
    cam_pose = get_camera_pose(pose)
    for x, y, z, x_im, y_im in plot_points:
        transformed_point = se3.apply(cam_pose, (x, y, z))
        tracked_points.append(transformed_point)
        marked_im = cv2.circle(marked_im, (x_im, y_im), 2, (255, 0, 0), 2)

    proj_lines = []
    print(lines[0])
    for line in lines:
        p1 = line[:2]
        p2 = line[2:]
        new_p1 = se3.apply(cam_pose, transform_point(p1))
        new_p2 = se3.apply(cam_pose, transform_point(p2))
        proj_lines.append((new_p1, new_p2))

    print(pose)
    scaled_lines = []
    for p1, p2 in proj_lines:
        p1 = map_scale(p1[:2])
        p2 = map_scale(p2[:2])
        scaled_lines.append((*p1, *p2))

    pose_x = pose['x']
    pose_y = pose['y']
    pose_center = (pose_x, pose_y)
    pose_px = map_scale(pose_center)
    pose_px_x, pose_px_y = pose_px

    max_depth = 2
    scale = map_scaling * max_depth
    zero_mask = np.zeros(map_img.shape[:2], dtype=np.uint8)
    circle_mask = zero_mask.copy()
    cv2.circle(circle_mask, pose_px, scale, (1, 1, 1), -1)
    map_thresh = map_img[:, :, 0] > 70
    centered_lines = []
    for x1, y1, x2, y2 in scaled_lines:
        centered_lines.append([x1 - pose_px_x, y1 - pose_px_y,
                               x2 - pose_px_x, y2 - pose_px_y])
    confidences = []
    #deltas = [-0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02]
    #deltas = [-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04]
    #deltas = [-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08]
    deltas = [-0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01,
              0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    for delta_angle in deltas:
        rot_mat = np.array([[np.cos(delta_angle), np.sin(delta_angle)],
                            [-np.sin(delta_angle), np.cos(delta_angle)]])
        transform_lines = []
        for line in centered_lines:
            transform_lines.append(np.array([*(rot_mat @ line[:2] + pose_px),
                                             *(rot_mat @ line[2:] + pose_px)],
                                             dtype=np.int32))
        lines_only = zero_mask.copy()
        plot_lines(lines_only, transform_lines, 1)
        lines_only *= circle_mask
        line_px = np.where(lines_only == 1)
        max_score = len(line_px[0])
        score = np.sum(map_thresh[line_px])
        confidences.append(score / max_score)
    if max(confidences) > 0.0:
        confidences = np.array(confidences)
        total = np.sum(confidences)
        net_delta = vo.dot(confidences, deltas) / total
        print("guess:", net_delta, confidences)
        estimated_rot_err -= net_delta
        pose['heading'] += estimated_rot_err
    else:
        print("bad data", confidences)

    points_angles = []
    heading = pose['heading']
    max_angle = heading + fov_x/2
    min_angle = heading - fov_x/2
    unit_heading = (np.cos(heading), np.sin(heading))
    # mmmmm rotation reversed due to left handed coord sys
    heading_perp = (np.sin(heading), np.cos(heading))
    for x1, y1, x2, y2 in scaled_lines:
        v1 = vo.sub((x1, y1), pose_px)
        v2 = vo.sub((x2, y2), pose_px)
        v1_l = vo.norm(v1)
        v2_l = vo.norm(v2)
        angle1 = np.arccos(vo.dot(heading_perp, v1) / v1_l) - np.pi/2
        angle2 = np.arccos(vo.dot(heading_perp, v2) / v2_l) - np.pi/2
        if v1_l > scale:
            v1 = vo.mul(v1, scale/ v1_l)
        if v2_l > scale:
            v2 = vo.mul(v2, scale/ v2_l)
        if abs(angle1) < fov_x / 2:
            points_angles.append((angle1, vo.add(v1, pose_px)))
        if abs(angle2) < fov_x / 2:
            points_angles.append((angle2, vo.add(v2, pose_px)))
    points_angles.sort()
    r_start = vo.norm(vo.sub(points_angles[0][1], pose_px))
    r_end = vo.norm(vo.sub(points_angles[-1][1], pose_px))
    # More rotation inversion
    point_start = vo.add((r_start * np.cos(min_angle), -r_start * np.sin(min_angle)), pose_px)
    point_end = vo.add((r_end * np.cos(max_angle), -r_end * np.sin(max_angle)), pose_px)
    observe_mask = np.zeros(map_img.shape[:2], dtype=np.uint8)
    observe_points = np.array([point_start] + [x[1] for x in points_angles] + [point_end, pose_px], dtype=np.int32)
    cv2.fillPoly(observe_mask, pts=[observe_points], color=1)
    cv2.imshow("mask", observe_mask*255)

    cam_inv = se3.inv(cam_pose)
    if prev_points is not None:
        #transform_lines = []
        #for p1, p2 in prev_lines:
        #    new_p1 = project_point(cam_inv, p1)
        #    new_p2 = project_point(cam_inv, p2)
        #    transform_lines.append([*new_p1, *new_p2])

        #plot_lines(deriv, transform_lines, 100)
        for point in prev_points:
            x_im, y_im = project_point(cam_inv, point)
            if x_im >= 0 and x_im < w and y_im >= 0 and y_im < h:
                marked_im = cv2.circle(marked_im, (x_im, y_im), 5, (0, 0, 255), 2)

    pointer_scale = 0.5
    disp_map = map_img.copy()
    plot_lines(disp_map, scaled_lines, (0, 0, 255))
    cv2.circle(disp_map, pose_px, 5, (0, 255, 0), 1)
    cv2.line(disp_map, pose_px,
                       map_scale((pose_x + pointer_scale * np.cos(heading),
                                  pose_y + pointer_scale * np.sin(heading))), (0, 255, 0))
    cv2.line(disp_map, pose_px,
                       map_scale((pose_x + 100*np.cos(max_angle),
                                  pose_y + 100*np.sin(max_angle))), (255, 0, 0))
    cv2.line(disp_map, pose_px,
                       map_scale((pose_x + 100*np.cos(min_angle),
                                  pose_y + 100*np.sin(min_angle))), (255, 0, 0))
    cv2.imshow("map", disp_map)
    #cv2.imshow("before", deriv / 200)
    cv2.imshow("marked", marked_im)
    key = cv2.waitKeyEx(0)
    if key == 65361:
        # left arrow
        if i > 0:
            i -= 1
    elif key == 27 or key == ord('q'):
        break
    else:
        i += 1
    new_lines = np.zeros(map_img.shape, dtype=np.uint8)
    plot_lines(new_lines, scaled_lines, (255, 255, 255))
    map_img -= np.array(map_img * observe_mask[:, :, np.newaxis] * 0.1, dtype=np.uint8)
    map_img += np.array(new_lines * circle_mask[:, :, np.newaxis], dtype=np.uint8)

    prev_pose = cam_pose
    prev_points = tracked_points
    prev_lines = proj_lines
    prev_observe_mask = observe_mask
