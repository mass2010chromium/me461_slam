#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <deque>
#include <ctime>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dbscan.hpp"

#include <motionlib/vectorops.h>
#include <motionlib/so3.h>
#include <motionlib/se3.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>

#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <simple-web-server/client_http.hpp>
#include <future>

#include <Eigen/Dense> 
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::VectorXd;

using namespace boost::property_tree;
using HttpClient = SimpleWeb::Client<SimpleWeb::HTTP>;

using std::vector;
using std::cout;
using std::endl;
using cv::Mat;

const motion_dtype z_real = -0.105;
struct CameraInfo {
    motion_dtype fx;
    motion_dtype fy;
    motion_dtype mid_x;
    motion_dtype mid_y;
    motion_dtype fov_x;
    motion_dtype fov_y;
};
typedef struct CameraInfo CameraInfo;

CameraInfo camera_info;

// Transform from pixel space into 3d space.
static inline void transform_point(vptr_r out, const vptr_r in) {
    out[0] = camera_info.fy * z_real / (camera_info.mid_y - in[1]);
    out[1] = (camera_info.mid_x - in[0]) * out[0] / camera_info.fx;
    out[2] = z_real;
}

// Project a point in real space into the given camera pose.
static inline void project_point(vptr_r out, const tptr_r cam_pose, const vptr_r point) {
    motion_dtype p[3];
    __se3_apply(p, cam_pose, point);
    out[0] = camera_info.mid_x - p[1] * camera_info.fx / p[0];
    out[1] = camera_info.mid_y - z_real * camera_info.fy / p[0];
}

// Get se3 pose of camera, given robot pose.
static inline void get_camera_pose(tptr_r ret, const vptr_r robot_pose) {
    static const motion_dtype axis[3] = {0, 0, 1};
    static const motion_dtype camera_pos[3] = {0.04, 0.0325, 0.105};  // Camera relative to robot center.
    motion_dtype tmp[3];
    tmp[0] = robot_pose[0];
    tmp[1] = robot_pose[1];
    tmp[2] = 0;
    double heading = robot_pose[2];
    __so3_rotation(ret, axis, heading);
    __so3_apply(ret+9, ret, camera_pos);
    __vo_add(ret+9, ret+9, tmp, 3);
}

#define line_t motion_dtype*
#define line_first(l) (l)
#define line_second(l) ((l)+2)

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
    if (area < 0) {
        // Point is opposite corner1 w.r.t. corner2.
        ret[0] = __vo_norm(pointvec, 2);
        ret[1] = -area / __vo_norm(linevec, 2);
        return;
    }
    motion_dtype pointvec2[2];
    __vo_subv(pointvec2, point, corner2, 2);
    if (__vo_dot(linevec, pointvec2, 2) > 0) {
        // Point is opposite corner2 w.r.t. corner1.
        ret[0] = __vo_norm(pointvec2, 2);
        ret[1] = area / __vo_norm(linevec, 2);
        return;
    }
    ret[0] = area / __vo_norm(linevec, 2);
    ret[1] = ret[0];
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
    return min_deviation*5 + max_deviation;
}

static inline cv::Point _Point(vptr i) {
    return cv::Point(i[0], i[1]);
}

void plot_lines(Mat& img, vector<vptr> lines, const cv::Scalar& color) {
    for (vptr line : lines) {
        cv::line(img, _Point(line_first(line)), _Point(line_second(line)), color);
    }
}

#define BUFFER_SIZE (2048*8)
motion_dtype alloc_buffer[BUFFER_SIZE];
int alloc_idx = 0;
line_t fast_alloc_vec(size_t sz) {
    if (alloc_idx + sz >= BUFFER_SIZE) {
        alloc_idx = 0;
    }
    else {
        alloc_idx += sz;
    }
    return alloc_buffer + alloc_idx;
}

vector<line_t> split_image(const Mat& grad_img, vector<line_t>& prev_transform_lines) {
    cv::Size size = grad_img.size();
    int h = size.height;
    int w = size.width;
    Mat prev_mask = Mat::zeros(size, CV_8UC1);
    plot_lines(prev_mask, prev_transform_lines, 1);
    Mat target = prev_mask.mul(grad_img);
    for (int j = 0; j < w; ++j) {
        for (int i = h-1; i > -1; --i) {
            if (grad_img.at<char>(i, j)) {
                target.at<char>(i, j) += 128;
            }
        }
    }

    // imshow(target)
    cv::imshow("target", target);
    vector<cv::Vec4i> match_lines;
    cv::HoughLinesP(target, match_lines, 5, 0.01, 50, 10, 10);
    vector<line_t> ret;
    if (match_lines.size() == 0) {
        cout << "No lines found..." << endl;
        return ret;
    }
    if (match_lines.size() > 50) {
        match_lines.resize(50);
    }
    target = 0;
    vector<line_t> data(match_lines.size());
    for (int i = 0; i < match_lines.size(); ++i) {
        auto& v = match_lines[i];
        line_t new_line = fast_alloc_vec(4);
        new_line[0] = v[0];
        new_line[1] = v[1];
        new_line[2] = v[2];
        new_line[3] = v[3];
        data[i] = new_line;
    }
    int num_clusters;
    vector<int> groups = dbscan(num_clusters, 125, 1, line_distance, data);
    vector<vector<vptr>> boxes;
    vector<line_t> plots;
    for (int i = 0; i < match_lines.size(); ++i) {
        boxes.push_back(vector<vptr>());
    }
    for (int i = 0; i < match_lines.size(); ++i) {
        if (groups[i] == -1) {
            plots.push_back(data[i]);
        }
        else {
            vptr p1 = line_first(data[i]);
            vptr p2 = line_second(data[i]);
            boxes[i].push_back(p1);
            boxes[i].push_back(p2);
        }
    }
    for (auto& box_points : boxes) {
        int n_points = box_points.size();
        Matrix<motion_dtype, Dynamic, 2> A(n_points, 2);
        Matrix<motion_dtype, Dynamic, 1> y_vals(n_points);
        motion_dtype min_x = box_points[0][0];
        motion_dtype max_x = box_points[0][0];
        for (int i = 0; i < n_points; ++i) {
            y_vals(i, 0) = box_points[i][1];
            auto x = box_points[i][0];
            A(i, 0) = x;
            A(i, 1) = 1;
            if (x < min_x) { min_x = x; }
            if (x > max_x) { max_x = x; }
        }

        // NOTE: numerical issues maybe
        auto least_square_soln = (A.transpose() * A).ldlt().solve(A.transpose() * y_vals);
        motion_dtype m = least_square_soln(1, 0);
        motion_dtype b = least_square_soln(2, 0);
        motion_dtype left_y = min_x * m + b;
        motion_dtype right_y = max_x * m + b;
        if (left_y >= 0 && left_y < h && right_y >= 0 && right_y < h) {
            line_t new_line = fast_alloc_vec(4);
            new_line[0] = min_x;
            new_line[1] = left_y;
            new_line[2] = max_x;
            new_line[3] = right_y;
            ret.push_back(new_line);
        }
    }
    return ret;
}

const double map_w = 400;
const double map_center = map_w/2;
const double map_scaling = 25;
static inline void map_scale(double* dest, double* src) {
    dest[0] = src[0] * map_scaling + map_center;
    dest[1] = -src[1] * map_scaling + map_center;
}

int main()
{
    std::ifstream t("calibration/intrinsics.json");
    std::stringstream buffer;
    buffer << t.rdbuf();
    ptree pt;
    read_json(buffer, pt);
    auto json_mat = pt.get_child("matrix");
    auto json_distort = pt.get_child("distortion");
    Mat camera_mat = Mat::zeros(3, 3, CV_64F);
    vector<double> distortion;

    for (const auto& _distort : json_distort) {
        for (const auto& node : _distort.second) {
            distortion.push_back(node.second.get_value<double>());
        }
    }

    int i = 0;
    for (const auto& row: json_mat) {
        int j = 0;
        for (const auto& node : row.second) {
            camera_mat.at<double>(i, j) = node.second.get_value<double>();
            ++j;
        }
        ++i;
    }
    std::cout << camera_mat << std::endl;

    camera_info.mid_x = camera_mat.at<double>(0, 2);
    camera_info.mid_y = camera_mat.at<double>(1, 2);
    camera_info.fx = camera_mat.at<double>(0, 0);
    camera_info.fy = camera_mat.at<double>(1, 1);
    camera_info.fov_x = pt.get<double>("fovx");
    camera_info.fov_y = pt.get<double>("fovy");

    cv::VideoCapture camera(0);
    cv::namedWindow("raw", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("target", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("grad", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("map", cv::WINDOW_AUTOSIZE);
    const double FEET_TO_METER = 0.3048;

    //double prev_pose[5];
    //vector<line_t> prev_points;
    vector<line_t> prev_lines;

    // map_w x map_w 3-channel 8-bit image, initial value 255
    Mat map_img(map_w, map_w, CV_8UC3, 255);
    Mat observe_mask(map_w, map_w, CV_8UC1);
    Mat disp_map(map_w, map_w, CV_8UC3);
    Mat circle_mask(map_w, map_w, CV_32FC1);
    Mat new_lines(map_w, map_w, CV_8UC3);

    double estimated_rot_err;

    motion_dtype* pose = NULL;
    double prev_head_tmp;
    std::deque<double*> pose_queue;
    const size_t VIDEO_DELAY = 1;

    Mat disp;
    Mat undistort;
    Mat deriv;

    for (;;) {
        auto start = std::chrono::system_clock::now();

        HttpClient client("localhost:8080");
        volatile bool pose_gotten = false;
        std::stringstream pose_info;
        client.request("GET", "/pose", "", [&pose_gotten, &pose_info](std::shared_ptr<HttpClient::Response> response, const SimpleWeb::error_code& ec) {
            if (!ec) {
                pose_info << response->content.rdbuf();
            }
            else {
                pose_info << R"({"x":0,"y":0,"heading":0,"v":0,"w":0})";
                std::cout << "http request failed" << std::endl;
            }
            pose_gotten = true;
        });
        client.io_service->run();
        bool res = camera.read(disp);
        //cv::cvtColor(disp, disp, cv::COLOR_BGR2GRAY);
        cv::undistort(disp, undistort, camera_mat, distortion);

        while (!pose_gotten) {};
        ptree pt;
        read_json(pose_info, pt);
        motion_dtype* new_pose = (motion_dtype*) malloc(5*sizeof(double));
        new_pose[0] = pt.get<double>("x");
        new_pose[1] = pt.get<double>("y");
        new_pose[2] = pt.get<double>("heading");
        new_pose[3] = pt.get<double>("v");
        new_pose[4] = pt.get<double>("w");

        if (pose == NULL) {
            pose = new_pose;
            prev_head_tmp = pose[2];
        }
        else {
            pose_queue.push_back(new_pose);
            bool free_pose = false;
            if (pose_queue.size() == VIDEO_DELAY + 1) {
                new_pose = pose_queue[0];
                pose_queue.pop_front();
                free_pose = true;
            }
            motion_dtype delta[2];
            __vo_subv(delta, new_pose, pose, 2);
            double _cos = cos(estimated_rot_err);
            double _sin = sin(estimated_rot_err);
            pose[0] += delta[0] * _cos + delta[1] * _sin;
            pose[1] += delta[1] * _cos - delta[0] * _sin;
            printf("raw pose: %f %f %f\n", new_pose[0], new_pose[1], new_pose[2]);
            printf("heading change %f\n", new_pose[2] - prev_head_tmp);
            prev_head_tmp = new_pose[2];
            pose[2] = new_pose[2] - estimated_rot_err;
            pose[3] = new_pose[3];
            pose[4] = new_pose[4];
            if (free_pose) {
                free(new_pose);
            }
        }
        
        disp = undistort.clone();

        int height = disp.size().height;
        Mat processing = undistort(cv::Range(height/2, height), cv::Range::all());
        cv::Canny(processing, deriv, 150, 300);
        deriv = deriv / 2;

        motion_dtype camera_pose[12];
        motion_dtype cam_inv[12];
        get_camera_pose(camera_pose, pose);
        __se3_inv(cam_inv, camera_pose);
        
        vector<line_t> prev_transform_lines;
        for (line_t line : prev_lines) {
            line_t new_line = fast_alloc_vec(4);
            project_point(line_first(new_line), cam_inv, line_first(line));
            project_point(line_second(new_line), cam_inv, line_second(line));
            prev_transform_lines.push_back(new_line);
        }

        vector<line_t> lines = split_image(deriv, prev_transform_lines);

        observe_mask = 0;
        disp_map = 0;
        new_lines = 0;
        vector<line_t> proj_lines;
        if (lines.size() > 0) {
            plot_lines(deriv, lines, 255);
            // plot tracked points
            
            vector<line_t> scaled_lines;
            for (line_t line : lines) {
                line[1] += height / 2;
                line[3] += height / 2;
                line_t proj_line = fast_alloc_vec(5);
                motion_dtype scratch[3];
                transform_point(scratch, line_first(line));
                __se3_apply(line_first(proj_line), camera_pose, scratch);
                transform_point(scratch, line_second(line));
                __se3_apply(line_second(proj_line), camera_pose, scratch);
                proj_lines.push_back(proj_line);

                line_t scaled_line = fast_alloc_vec(4);
                map_scale(line_first(scaled_line), line_first(proj_line));
                map_scale(line_second(scaled_line), line_second(proj_line));
                scaled_lines.push_back(scaled_line);
            }

            motion_dtype pose_x = pose[0];
            motion_dtype pose_y = pose[1];
            motion_dtype heading = pose[2];
            motion_dtype ch = cos(heading);
            motion_dtype sh = sin(heading);
            motion_dtype unit_heading[2] = { ch, -sh }; // funky rotation flipping due to image coords
            motion_dtype heading_perp[2] = { sh, ch }; // funky rotation flipping due to image coords
            motion_dtype max_angle = heading + camera_info.fov_x/2;
            motion_dtype min_angle = heading - camera_info.fov_x/2;
            motion_dtype pose_center[2] = { pose_x, pose_y };
            motion_dtype pose_px[2];
            map_scale(pose_px, pose_center);
            cv::Point _pose_px = _Point(pose_px);

            const double max_depth = 1;
            const double scale = map_scaling * max_depth;
            circle_mask = 0;
            cv::circle(circle_mask, _pose_px, scale*3, 0.25, -1);
            cv::circle(circle_mask, _pose_px, scale*2, 0.5, -1);
            cv::circle(circle_mask, _pose_px, scale*1, 1, -1);

            vector<vptr> points_angles;
            for (line_t line : scaled_lines) {
                motion_dtype v1[2]; __vo_subv(v1, line_first(line), pose_px, 2);
                motion_dtype v2[2]; __vo_subv(v2, line_second(line), pose_px, 2);
                motion_dtype v1_l = __vo_norm(v1, 2);
                motion_dtype v2_l = __vo_norm(v2, 2);
                motion_dtype angle1 = acos(__vo_dot(heading_perp, v1, 2) / v1_l) - (M_PI/2);
                motion_dtype angle2 = acos(__vo_dot(heading_perp, v2, 2) / v2_l) - (M_PI/2);
                if (fabs(angle1) < camera_info.fov_x/2 && __vo_dot(unit_heading, v1, 2) > 0) {
                    vptr tmp = fast_alloc_vec(3);
                    tmp[0] = angle1;
                    __vo_add(tmp+1, v1, pose_px, 2);
                    points_angles.push_back(tmp);
                }
                if (fabs(angle2) < camera_info.fov_x/2 && __vo_dot(unit_heading, v2, 2) > 0) {
                    vptr tmp = fast_alloc_vec(3);
                    tmp[0] = angle2;
                    __vo_add(tmp+1, v2, pose_px, 2);
                    points_angles.push_back(tmp);
                }
            }
            if (points_angles.size() > 0) {
                std::sort(points_angles.begin(), points_angles.end(), line_cmp);
                motion_dtype start[2];
                motion_dtype end[2];
                __vo_subv(start, points_angles[0] + 1, pose_px, 2);
                motion_dtype r_start = __vo_norm(start, 2);
                __vo_subv(end, points_angles[points_angles.size() - 1] + 1, pose_px, 2);
                motion_dtype r_end = __vo_norm(end, 2);
                // More flipped angle garbage
                start[0] = r_start * cos(min_angle);
                start[1] = -r_start * sin(min_angle);
                __vo_add(start, start, pose_px, 2);
                end[0] = r_end * cos(max_angle);
                end[1] = -r_end * sin(max_angle);
                __vo_add(end, end, pose_px, 2);

                vector<cv::Point> polygon_points;
                polygon_points.push_back(_Point(start));
                for (vptr p : points_angles) {
                    polygon_points.push_back(_Point(p));
                }
                polygon_points.push_back(_Point(end));
                polygon_points.push_back(_pose_px);
                cv::fillPoly(observe_mask, polygon_points, 1);
                observe_mask *= circle_mask;
            }
            // imshow observe mask
            plot_lines(disp_map, scaled_lines, {0, 0, 255});
            plot_lines(new_lines, scaled_lines, {255, 255, 255});
        }
        
        // if prev points not none:
        //     plot the tracking points

        motion_dtype pose_x = pose[0];
        motion_dtype pose_y = pose[1];
        motion_dtype heading = pose[2];
        motion_dtype max_angle = heading + camera_info.fov_x/2;
        motion_dtype min_angle = heading - camera_info.fov_x/2;
        motion_dtype pose_center[2] = { pose_x, pose_y };
        motion_dtype pose_px[2];
        map_scale(pose_px, pose_center);
        
        motion_dtype pointer_scale = 0.5;
        cv::Point center(pose_px[0], pose_px[1]);
        cv::circle(disp_map, center, 5, {0, 255, 0}, 1);

        double leader[2] = {pose_x + pointer_scale * cos(heading),
                            pose_y + pointer_scale * sin(heading)};
        map_scale(leader, leader);
        cv::line(disp_map, center, _Point(leader), {0, 255, 0});
        double max_pt[2] = {pose_x + 100 * cos(max_angle),
                            pose_y + 100 * sin(max_angle)};
        map_scale(max_pt, max_pt);
        cv::line(disp_map, center, _Point(max_pt), {255, 0, 0});
        double min_pt[2] = {pose_x + 100 * cos(min_angle),
                            pose_y + 100 * sin(min_angle)};
        map_scale(min_pt, max_pt);
        cv::line(disp_map, center, _Point(max_pt), {255, 0, 0});

        cv::imshow("raw", disp);
        cv::imshow("grad", deriv);
        cv::imshow("map", disp_map);
        int key = cv::waitKeyEx(1);
        if (key == 27 || key == 'q') {
            break;
        }

        Mat tmp = observe_mask * 0.05;
        cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);
        tmp = tmp.mul(map_img);
        map_img -= tmp;
        Mat tmp2 = observe_mask.clone();
        cv::cvtColor(tmp2, tmp2, cv::COLOR_GRAY2BGR);
        tmp2 = tmp2.mul(new_lines);
        map_img += tmp2;

        prev_lines = proj_lines;

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "SPF: " << elapsed_seconds.count() <<  std::endl;
    }
    return 0;
}
