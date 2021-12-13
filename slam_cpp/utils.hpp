#pragma once

#include <math.h>
#include <iostream>

#include "types.h"
#include <motionlib/vectorops.h>
#include <motionlib/so3.h>
#include <motionlib/se3.h>

// https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
// Too lazy to think
static inline bool line_intersection(vptr ret, line_t l1, line_t l2) {
    motion_dtype x1 = l1[0];
    motion_dtype y1 = l1[1];
    motion_dtype x2 = l1[2];
    motion_dtype y2 = l1[3];
    motion_dtype x3 = l2[0];
    motion_dtype y3 = l2[1];
    motion_dtype x4 = l2[2];
    motion_dtype y4 = l2[3];

    motion_dtype D = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
    if (fabs(D) < 1e-5) {
        return false;
    }
    ret[0] = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / D;
    ret[1] = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / D;
    motion_dtype tmp;
    if (x1 > x2) {
        tmp = x2;
        x2 = x1;
        x1 = tmp;
    }
    if (y1 > y2) {
        tmp = y2;
        y2 = y1;
        y1 = tmp;
    }
    if (x3 > x4) {
        tmp = x4;
        x4 = x3;
        x3 = tmp;
    }
    if (y3 > y4) {
        tmp = y4;
        y4 = y3;
        y3 = tmp;
    }
    if (x1 > ret[0] || x2 < ret[0] || x3 > ret[0] || x4 < ret[0]) {
        return false;
    }
    if (y1 > ret[1] || y2 < ret[1] || y3 > ret[1] || y4 < ret[1]) {
        return false;
    }
    return true;
}

static inline motion_dtype normalize_angle(motion_dtype angle) {
    angle = fmod(angle, 2*M_PI);
    if(angle < 0) {
        angle += 2*M_PI;
    }
    return angle;
}

// Inputs must be normalized.
// Computes b - a.
static inline motion_dtype angle_distance_signed(motion_dtype a, motion_dtype b) {
    if (b > a) {
        motion_dtype forward = b - a;
        if (forward < M_PI) { return forward; }
        return forward - 2*M_PI;
    }
    if (b < a) {
        motion_dtype backward = a - b;
        if (backward < M_PI) { return -backward; }
        return 2*M_PI - backward;
    }
    return 0;
}

// Inputs must be normalized!
static inline motion_dtype angle_distance(motion_dtype a, motion_dtype b) {
    motion_dtype max_angle = a;
    motion_dtype min_angle = b;
    if (b > a) {
        max_angle = b;
        min_angle = a;
    }
    motion_dtype err = max_angle - min_angle;
    motion_dtype err2 = 2*M_PI - err;
    if (err < err2) { return err; }
    return err2;
}

bool line_cmp(line_t i, line_t j) { return (i[0] < j[0]); }

// Get distance between point and line, in two ways.
// First value is the distance to segment.
// Second value is distance to the infinite line.
static inline void point_to_line(vptr_r ret, const vptr_r point, const line_t line) {
    const vptr corner1 = line_first(line);
    const vptr corner2 = line_second(line);
    motion_dtype linevec[2];
    motion_dtype pointvec[2];
    __vo_subv(linevec, corner2, corner1, 2);
    __vo_subv(pointvec, point, corner1, 2);
    motion_dtype area = __vo_dot(linevec, pointvec, 2);
    motion_dtype line_dist = fabs(__vo_cross2(linevec, pointvec)) / __vo_norm(linevec, 2);
    ret[1] = line_dist;
    if (area < 0) {
        // Point is opposite corner1 w.r.t. corner2.
        ret[0] = __vo_norm(pointvec, 2);
        return;
    }
    motion_dtype pointvec2[2];
    __vo_subv(pointvec2, point, corner2, 2);
    if (__vo_dot(linevec, pointvec2, 2) > 0) {
        // Point is opposite corner2 w.r.t. corner1.
        ret[0] = __vo_norm(pointvec2, 2);
        return;
    }
    ret[0] = line_dist;
}

motion_dtype line_distance(const line_t l1, const line_t l2) {
    motion_dtype retvals[8];
    point_to_line(retvals + 0, line_first(l1), l2);
    point_to_line(retvals + 2, line_second(l1), l2);
    point_to_line(retvals + 4, line_first(l2), l1);
    point_to_line(retvals + 6, line_second(l2), l1);
    motion_dtype min_deviation = retvals[0];
    motion_dtype max_deviation = retvals[1];
    for (int i = 2; i < 8; i += 2) {
        if (retvals[i] < min_deviation) { min_deviation = retvals[i]; }
        if (retvals[i+1] > max_deviation) { max_deviation = retvals[i+1]; }
    }
    motion_dtype angle1 = normalize_angle(atan2(l1[3] - l1[1], l1[2] - l1[0]));
    motion_dtype angle2 = normalize_angle(atan2(l2[3] - l2[1], l2[2] - l2[0]));
    motion_dtype angle_err = angle_distance(angle1, angle2);
    if (angle_err > M_PI/2) {
        angle_err = M_PI - angle_err;
    }
    //motion_dtype scratch[2];
    //__vo_subv(line_first(l1), line_second(l1), 2);
    //motion_dtype length = __vo_norm(scratch, 2);
    //__vo_subv(line_first(l2), line_second(l2), 2);
    //motion_dtype length2 = __vo_norm(scratch, 2);
    //if (length2 < length) { length = length2; }
    return (min_deviation * (1 + angle_err)
            + angle_err * (1 + min_deviation));
}

void linspace(vptr ret, motion_dtype center, motion_dtype spread, int step) {
    motion_dtype spread_step = spread / step;
    for (int i = 0; i < step; ++i) {
        ret[i] = center - spread_step * (step - i);
    }
    ret[step] = center;
    for (int i = 0; i < step; ++i) {
        ret[i + step] = center + spread_step * (i - step);
    }
}
