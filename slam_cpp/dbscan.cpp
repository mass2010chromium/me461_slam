#include "dbscan.h"

#include "headers.hpp"

#include <deque>
#include <assert.h>

#include "types.h"
#include <motionlib/vectorops.h>
#include "utils.hpp"

#include "fast_alloc.h"

#include <iostream>
#include <stdio.h>
#include <utility>
using std::pair;

//#include <Eigen/Dense> 
//using Eigen::Matrix;
//using Eigen::Dynamic;
//using Eigen::VectorXd;

using std::deque;
using std::cout;
using std::endl;

// return: groups. (-1 for noise)
// uses brute force search lol
// https://en.wikipedia.org/wiki/DBSCAN#Algorithm
vector<int> dbscan(int& num_clusters, motion_dtype eps, int min_samples,
                   motion_dtype (metric)(const vptr, const vptr), vector<vptr>& data) {
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

void project_line(line_t ret, line_t src, vector<vptr>& points) {
    motion_dtype line_zero[2] = {src[0], src[1]};
    motion_dtype line_one[2]; __vo_subv(line_one, line_second(src), line_first(src), 2);
    __vo_unit(line_one, line_one, 1e-5, 2);
    motion_dtype b_aligned = __vo_dot(line_zero, line_one, 2);
    __vo_madd(line_zero, line_zero, line_one, -b_aligned, 2);
    motion_dtype min_dir = __vo_dot(points[0], line_one, 2);
    motion_dtype max_dir = min_dir;
    for (int i = 1; i < points.size(); ++i) {
        auto point = points[i];
        motion_dtype res = __vo_dot(point, line_one, 2);
        if (res < min_dir) { min_dir = res; }
        if (res > max_dir) { max_dir = res; }
    }
    __vo_madd(line_first(ret), line_zero, line_one, min_dir, 2);
    __vo_madd(line_second(ret), line_zero, line_one, max_dir, 2);
}

void merge_lines(line_t ret, vector<vptr>& points) {
    int n_points = points.size();
//    Matrix<motion_dtype, Dynamic, 2> A(n_points, 2);
//    Eigen::DiagonalMatrix<motion_dtype, Dynamic> W(n_points);
//    for (int i = 0; i < n_points; ++i) {
//        W.diagonal()[i] = points[i].second;
//        printf("%f ", points[i].second);
//    }
//    Matrix<motion_dtype, Dynamic, 1> y_vals(n_points);
    //printf("begin\n");
    motion_dtype x_sum = 0;
    motion_dtype y_sum = 0;
    motion_dtype x2_sum = 0;
    motion_dtype y2_sum = 0;
    motion_dtype xy_sum = 0;
    for (int i = 0; i < n_points; ++i) {
        auto x = points[i][0];
        auto y = points[i][1];
        //printf("%f %f\n", x, y);
        x_sum += x;
        y_sum += y;
        x2_sum += x*x;
        y2_sum += y*y;
        xy_sum += x*y;
//        y_vals(i, 0) =y;
//        A(i, 0) = x;
//        A(i, 1) = 1;
    }

//    auto least_square_soln = (A.transpose()*W*A).ldlt().solve(A.transpose()*y_vals);
//    motion_dtype m = least_square_soln(0, 0);
//    motion_dtype b = least_square_soln(1, 0);
    motion_dtype x_bar = x_sum/n_points;
    motion_dtype y_bar = y_sum/n_points;
    // https://mathworld.wolfram.com/LeastSquaresFittingPerpendicularOffsets.html
    motion_dtype B = 0.5 * (y2_sum - n_points*y_bar*y_bar - x2_sum + n_points*x_bar*x_bar)
                    / (n_points*x_bar*y_bar - xy_sum);
    motion_dtype m1 = -B + sqrt(B*B+1);
    motion_dtype b1 = y_bar - m1*x_bar;
    motion_dtype m2 = -B - sqrt(B*B+1);
    motion_dtype b2 = y_bar - m2*x_bar;
    motion_dtype e1 = 0;
    motion_dtype e2 = 0;
    motion_dtype perp1[2] = {-m1, 1};
    __vo_unit(perp1, perp1, 1e-7, 2);
    motion_dtype perp2[2] = {-m2, 1};
    __vo_unit(perp2, perp2, 1e-7, 2);
    motion_dtype scratch[2];
    for (int i = 0; i < n_points; ++i) {
        scratch[0] = points[i][0]; scratch[1] = points[i][1] - b1;
        e1 += fabs(__vo_dot(scratch, perp1, 2));
        scratch[1] = points[i][1] - b2;
        e2 += fabs(__vo_dot(scratch, perp2, 2));
    }
    //printf("e1 %f e2 %f\n", e1, e2);
    motion_dtype m = m1;
    motion_dtype b = b1;
    motion_dtype _m = m2;
    motion_dtype _b = b2;
    if (e1 > e2) {
        m = m2;
        b = b2;
        _m = m1;
        _b = b1;
    }

    //printf("least squares: %fx + %f\n", m, b);
    //printf("B: %f, reject: %fx + %f\n", B, _m, _b);
    motion_dtype match_line[4] = {0, b, 1, m+b};
    project_line(ret, match_line, points);
    //printf("final %f: %f %f %f %f\n", __vo_dot(line_one, line_zero, 2), ret[0], ret[1], ret[2], ret[3]);
}

vector<line_t> dbscan_filter_lines(vector<line_t>& lines, vector<vptr>& existings,
                                   motion_dtype eps, int match_age) {
    int new_line_size = lines.size();
    for (auto line : existings) {
        lines.push_back(line);
    }
    //for (auto line : lines) {
    //    printf("%f %f, %f %f [%f]\n", line[0], line[1], line[2], line[3], line[4]);
    //}

    int num_clusters;
    vector<int> groups = dbscan(num_clusters, eps, 1, line_distance, lines);
    //for (auto group : groups) {
    //    cout << group << " ";
    //}
    //cout << endl;

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
                else {
                    //cout << "old death" << endl;
                }
            }
        }
        else {
            vptr p1 = line_first(lines[i]);
            vptr p2 = line_second(lines[i]);
            motion_dtype age;
            if (i < new_line_size) {
                age = 1;
            }
            else {
                age = lines[i][4];
            }
            for (int j = 0; j < age; ++j) {
                // TODO jank af weighted method
                boxes[groups[i]].push_back(p1);
                boxes[groups[i]].push_back(p2);
            }
            if (i >= new_line_size) {
                ages[groups[i]] = match_age;
                //cout << "old match" << endl;
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
    //int x;
    //std::cin >> x;
    return ret;
}

vector<line_t> dbscan_extend_lines(vector<line_t>& lines, vector<vptr>& existings,
                                   motion_dtype eps, int match_age) {
    int new_line_size = lines.size();
    for (auto line : existings) {
        lines.push_back(line);
    }
    //for (auto line : lines) {
    //    printf("%f %f, %f %f [%f]\n", line[0], line[1], line[2], line[3], line[4]);
    //}

    int num_clusters;
    vector<int> groups = dbscan(num_clusters, eps, 1, line_distance, lines);
    //for (auto group : groups) {
    //    cout << group << " ";
    //}
    //cout << endl;

    vector<line_t> ret;
    vector<pair<vector<vptr>, vector<vptr>>> boxes;
    vector<int> ages;
    for (int i = 0; i < num_clusters; ++i) {
        boxes.push_back(std::make_pair(vector<vptr>(), vector<vptr>()));
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
                else {
                    //cout << "old death" << endl;
                }
            }
        }
        else {
            vptr p1 = line_first(lines[i]);
            vptr p2 = line_second(lines[i]);
            motion_dtype age;
            if (i < new_line_size) {
                boxes[groups[i]].second.push_back(p1);
                boxes[groups[i]].second.push_back(p2);
            }
            else {
                boxes[groups[i]].first.push_back(p1);
                boxes[groups[i]].first.push_back(p2);
            }
            if (i >= new_line_size) {
                ages[groups[i]] = match_age;
                //cout << "old match" << endl;
            }
        }
    }
    for (int i = 0; i < boxes.size(); ++i) {
        auto& old_points = boxes[i].first;
        auto& new_points = boxes[i].second;
        vptr new_line;
        if (old_points.size() == 0) {
            if (new_points.size() == 0) { continue; }
            new_line = fast_alloc_vec(5);
            merge_lines(new_line, new_points);
        }
        else {
            motion_dtype old_line[4];
            merge_lines(old_line, old_points);
            new_points.push_back(line_first(old_line));
            new_points.push_back(line_second(old_line));
            new_line = fast_alloc_vec(5);
            project_line(new_line, old_line, new_points);
        }
        new_line[4] = ages[i];
        ret.push_back(new_line);
    }
    //int x;
    //std::cin >> x;
    return ret;
}
