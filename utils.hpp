#pragma once

#include <math.h>

#include "types.h"
#include <motionlib/vectorops.h>
#include <motionlib/so3.h>
#include <motionlib/se3.h>

static inline motion_dtype normalize_angle(motion_dtype angle) {
    angle = fmod(angle, 2*M_PI);
    if(angle < 0) {
        angle += 2*M_PI;
    }
    return angle;
}

// Inputs must be normalized!
static inline motion_dtype angle_distance(motion_dtype a, motion_dtype b) {
    motion_dtype max_angle = a;
    motion_dtype min_angle = b;
    if (b > a) {
        max_angle = b;
        min_angle = a;
    }
    motion_dtype err = max_angle = min_angle;
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
    return min_deviation * (1 + angle_err) + angle_err * (1 + min_deviation);
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
