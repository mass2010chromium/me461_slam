#pragma once

#include <math.h>
#include <stdio.h>

#include "types.h"
#include <motionlib/vectorops.h>

/*
 * saved_subset must be nonempty.
 */
motion_dtype loss(vector<line_t>& match_subset, vector<line_t>& saved_subset,
                  motion_dtype t, motion_dtype dx, motion_dtype dy) {
    motion_dtype r00 = cos(t);
    motion_dtype r10 = sin(t);
    motion_dtype r01 = -sin(t);
    motion_dtype r11 = r00;
    motion_dtype score = 0;
	for (line_t to_move : match_subset) {
        motion_dtype moved[4];
        moved[0] = r00*to_move[0] + r01*to_move[1] + dx;
        moved[1] = r10*to_move[0] + r11*to_move[1] + dy;
        moved[2] = r00*to_move[2] + r01*to_move[3] + dx;
        moved[3] = r10*to_move[2] + r11*to_move[3] + dy;

        line_t match = saved_subset[0];
        motion_dtype closest_dist = line_distance(moved, match);
        for (int i = 1; i < saved_subset.size(); ++i) {
            match = saved_subset[i];
            motion_dtype dist = line_distance(moved, match);
            if (dist < closest_dist) {
                closest_dist = dist;
            }
        }
        if (closest_dist > 0.3) {
            score += 0.3;
            continue;
        }
        score += closest_dist;
    }
    motion_dtype tmp[2];
    tmp[0] = dx;
    tmp[1] = dy;
    return score + __vo_norm(tmp, 2) + fabs(t);
}

const motion_dtype THETA_SPREAD_INIT = 0.3;
const motion_dtype DELTA_SPREAD_INIT = 0.05;
const int THETA_STEPS = 2;
const int DELTA_STEPS = 2;
const int REFINE_STEPS = 4;
motion_dtype register_lines(vptr ret, vector<line_t>& match_subset, vector<line_t>& saved_subset) {
    motion_dtype thetas[THETA_STEPS];
    motion_dtype delta_x[DELTA_STEPS];
    motion_dtype delta_y[DELTA_STEPS];
    motion_dtype theta_spread = THETA_SPREAD_INIT;
    motion_dtype delta_spread = DELTA_SPREAD_INIT;
    ret[0] = 0;
    ret[1] = 0;
    ret[2] = 0;
    motion_dtype best_loss = loss(match_subset, saved_subset, 0, 0, 0);
    for (int i = 0; i < REFINE_STEPS; ++i) {
        linspace(thetas, ret[0], theta_spread, THETA_STEPS);
        linspace(delta_x, ret[1], delta_spread, DELTA_STEPS);
        linspace(delta_y, ret[2], delta_spread, DELTA_STEPS);
        motion_dtype prev_loss = best_loss;
        for (int i_t = 0; i_t < THETA_STEPS; ++i_t) {
            motion_dtype t = thetas[i_t];
            for (int i_x = 0; i_x < DELTA_STEPS; ++i_x) {
                motion_dtype x = delta_x[i_x];
                for (int i_y = 0; i_y < DELTA_STEPS; ++i_y) {
                    motion_dtype y = delta_y[i_y];
                    motion_dtype calc_loss = loss(match_subset, saved_subset, t, x, y);
                    if (calc_loss < best_loss) {
                        ret[0] = t;
                        ret[1] = x;
                        ret[2] = y;
                        best_loss = calc_loss;
                    }
                }
            }
        }
        printf("refining... (%f, %f, %f) %f\n", ret[0], ret[1], ret[2], best_loss);
    }
    printf("Computed transform: (%f, %f, %f) %f\n", ret[0], ret[1], ret[2], best_loss);
    return best_loss;
}
