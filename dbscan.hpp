#pragma once

#include <vector>
#include <deque>

#include "types.h"
#include <motionlib/vectorops.h>
#include "utils.hpp"

#include <Eigen/Dense> 
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::VectorXd;

using std::vector;
using std::deque;

static inline deque<int> range_query(vector<vptr>& data,
                                     motion_dtype (metric)(const vptr, const vptr),
                                     int point, motion_dtype eps) {
    deque<int> ret;
    vptr p1 = data[point];
    for (unsigned int i = 0; i < data.size(); ++i) {
        if (metric(p1, data[i]) < eps) {
            ret.push_back(i);
        }
    }
    return ret;
}

// return: groups. (-1 for noise)
// uses brute force search lol
// https://en.wikipedia.org/wiki/DBSCAN#Algorithm
vector<int> dbscan(int& num_clusters, motion_dtype eps, int min_samples,
                   motion_dtype (metric)(const vptr, const vptr), vector<vptr> data) {
    // -2: undef
    // -1: noise
    // >0: cluster
    vector<int> labels(data.size(), -2);
    int current_cluster = 0;
    for (unsigned int i = 0; i < data.size(); ++i) {
        if (labels[i] != -2) { continue; }
        deque<int> neighbors = range_query(data, metric, i, eps);
        if (neighbors.size() < min_samples) {
            labels[i] = -1;
            continue;
        }
        labels[i] = current_cluster;
        while (!neighbors.empty()) {
            int j = neighbors[0];
            neighbors.pop_front();
            if (labels[j] == -1) { labels[j] = current_cluster; }
            if (labels[j] != -2) { continue; }
            labels[j] = current_cluster;
            deque<int> more_neighbors = range_query(data, metric, j, eps);
            if (more_neighbors.size() >= min_samples) {
                neighbors.insert(neighbors.end(), more_neighbors.begin(), more_neighbors.end());
            }
        }
        ++current_cluster;
    }
    num_clusters = current_cluster;
    return labels;
}

void merge_lines(line_t ret, vector<vptr>& points) {
    int n_points = points.size();
    Matrix<motion_dtype, Dynamic, 2> A(n_points, 2);
    Matrix<motion_dtype, Dynamic, 1> y_vals(n_points);
    motion_dtype min_x = points[0][0];
    motion_dtype max_x = points[0][0];
    motion_dtype min_y = points[0][1];
    motion_dtype max_y = points[0][1];
    for (int i = 0; i < n_points; ++i) {
        y_vals(i, 0) = points[i][1];
        auto x = points[i][0];
        auto y = points[i][1];
        A(i, 0) = x;
        A(i, 1) = 1;
        if (x < min_x) { min_x = x; }
        if (x > max_x) { max_x = x; }
        if (y < min_y) { min_y = y; }
        if (y > max_y) { max_y = y; }
    }

    auto least_square_soln = (A.transpose() * A).ldlt().solve(A.transpose() * y_vals);
    motion_dtype m = least_square_soln(0, 0);
    if (m > 0) {
        ret[0] = min_x;
        ret[1] = min_y;
        ret[2] = max_x;
        ret[3] = max_y;
    }
    else {
        ret[0] = min_x;
        ret[1] = max_y;
        ret[2] = max_x;
        ret[3] = min_y;
    }
}

vector<line_t> dbscan_filter_lines(vector<line_t>& lines, vector<vptr> existings, motion_dtype eps) {
    int new_line_size = lines.size();
    for (auto line : existings) {
        lines.push_back(line);
    }

    int num_clusters;
    vector<int> groups = dbscan(num_clusters, eps, 1, line_distance, lines);

    vector<line_t> ret;
    vector<vector<vptr>> boxes;
    vector<int> ages;
    for (int i = 0; i < num_clusters; ++i) {
        boxes.push_back(vector<vptr>());
        ages.push_back(1);
    }
    vector<int> groupsizes(num_clusters, 0);
    for (int group : groups) {
        if (group != -1) {
            ++groupsizes[group];
        }
    }
    for (int i = 0; i < lines.size(); ++i) {
        if (groups[i] == -1 || groupsizes[groups[i]] == 1) {
            vptr l = fast_alloc_vec(5);
            memcpy(l, lines[i], 5*sizeof(motion_dtype));
            if (i < new_line_size) {
                l[4] = 1;
                ret.push_back(l);
            }
            else {
                // NOTE: DO NOT REUSE MEMORY! it is important in this case that
                // the returned array has "fresh" line pointers, or they will expire!
                --l[4];
                if (l[4] > 0) {
                    ret.push_back(l);
                }
            }
        }
        else {
            vptr p1 = line_first(lines[i]);
            vptr p2 = line_second(lines[i]);
            boxes[i].push_back(p1);
            boxes[i].push_back(p2);
            if (i >= new_line_size) {
                ages[i] = 8;
            }
        }
    }
    for (int i = 0; i < boxes.size(); ++i) {
        auto box_points = boxes[i];
        if (box_points.size() == 0) { continue; }
        vptr new_line = fast_alloc_vec(5);
        merge_lines(new_line, box_points);
        new_line[4] = ages[i];
        ret.push_back(new_line);
    }
    return ret;
}
