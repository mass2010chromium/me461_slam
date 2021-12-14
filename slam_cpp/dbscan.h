#pragma once

#include "headers.hpp"

#include <deque>
#include <assert.h>

#include "types.h"
#include <motionlib/vectorops.h>
#include "utils.hpp"

#include <iostream>
#include <stdio.h>
#include <utility>
using std::pair;

#include <Eigen/Dense> 
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::VectorXd;

using std::deque;
using std::cout;
using std::endl;

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
                   motion_dtype (metric)(const vptr, const vptr), vector<vptr>& data);

void merge_lines(line_t ret, vector<pair<vptr, motion_dtype>>& points);

vector<line_t> dbscan_filter_lines(vector<line_t>& lines, vector<vptr>& existings, motion_dtype eps, int match_age);
