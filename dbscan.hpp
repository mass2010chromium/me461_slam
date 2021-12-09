#include <vector>
#include <deque>

#include <motionlib/vectorops.h>

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
