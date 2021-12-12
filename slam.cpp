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
#include <sys/mman.h>

#include "types.h"
#include "utils.hpp"
#include "fast_alloc.hpp"
#include "dbscan.hpp"
#include "optimize.hpp"

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

static inline cv::Point _Point(vptr i) {
    return cv::Point(i[0], i[1]);
}

motion_dtype map_w;
motion_dtype map_center;
motion_dtype map_scaling;
static inline void map_scale(vptr dest, vptr src) {
    dest[0] = src[0] * map_scaling + map_center;
    dest[1] = -src[1] * map_scaling + map_center;
}

void plot_lines(Mat& img, vector<vptr> lines, const cv::Scalar& color) {
    for (vptr line : lines) {
        cv::line(img, _Point(line_first(line)), _Point(line_second(line)), color);
    }
}

void draw_robot(Mat& img, vptr pose, motion_dtype pointer_scale=0.5) {
    motion_dtype pose_px[2];
    map_scale(pose_px, pose);
    motion_dtype heading = pose[2];
    motion_dtype min_angle = heading - camera_info.fov_x/2;
    motion_dtype max_angle = heading + camera_info.fov_x/2;
    auto pose_px_pt = _Point(pose_px);
    cv::circle(img, pose_px_pt, 5, {0, 255, 0}, 1);
    motion_dtype scratch[2];
    scratch[0] = pose_px[0] + pointer_scale * cos(heading);
    scratch[1] = pose_px[1] + pointer_scale * sin(heading);
    map_scale(scratch, scratch);
    cv::line(img, pose_px_pt, _Point(scratch), {0, 255, 0});

    const double fov_scale = 100;
    scratch[0] = pose_px[0] + fov_scale * cos(min_angle);
    scratch[1] = pose_px[1] + fov_scale * sin(min_angle);
    map_scale(scratch, scratch);
    cv::line(img, pose_px_pt, _Point(scratch), {255, 0, 0});
    scratch[0] = pose_px[0] + fov_scale * cos(max_angle);
    scratch[1] = pose_px[1] + fov_scale * sin(max_angle);
    map_scale(scratch, scratch);
    cv::line(img, pose_px_pt, _Point(scratch), {255, 0, 0});
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
                break;
            }
        }
    }

    // imshow(target)
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
    return data;
}

char* map_file(std::string filename, int open_flags, size_t map_size) {
    cout << "Mapping file " << filename << endl;
    int mmap_file = open(filename.c_str(), open_flags, 0777);
    if (mmap_file == -1) {
        perror("open mmap file failure");
        return NULL;
    }
    ftruncate(mmap_file, map_size);
    char* ret = (char*) mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED_VALIDATE, mmap_file, 0);
    if (ret == NULL) {
        perror("mmap failure");
        return NULL;
    }
    return ret;
}

/*
 * Convenience function for reading json from filename.
 */
void read_json_fname(const std::string fname, ptree& pt) {
    std::ifstream t(fname);
    std::stringstream buffer;
    buffer << t.rdbuf();
    read_json(buffer, pt);
}

void read_intrinsics(const std::string intrinsics_fname,
                     vector<double>& distortion, Mat& camera_mat, CameraInfo& camera_info) {
    ptree pt;
    read_json_fname(intrinsics_fname, pt);
    auto json_mat = pt.get_child("matrix");
    auto json_distort = pt.get_child("distortion");

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
}

int main(int argc, char** argv)
{
    Mat camera_mat = Mat::zeros(3, 3, CV_64F);
    vector<double> distortion;
    read_intrinsics("calibration/intrinsics.json", distortion, camera_mat, camera_info);
    ptree map_json;
    read_json_fname("map_info.json", map_json);
    map_w = map_json.get<double>("map_size");
    map_center = map_w / 2;
    map_scaling = map_json.get<double>("map_scale");

    cv::VideoCapture camera;
    char* image_buffer;
    char* map_buffer;
    map_buffer = map_file(".slam.map", O_RDWR | O_CREAT, 1000000);
    if (map_buffer == NULL) {
        return 1;
    }
    bool camera_mode = false;
    bool display_images = true;
    if (argc > 1 && std::string(argv[1]) == "--camera") {
        camera_mode = true;
        camera = cv::VideoCapture(0);
        camera.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }
    else {
        image_buffer = map_file(".webserver.video", O_RDWR, 1000000);
        if (image_buffer == NULL) {
            return 1;
        }
        if (argc > 1) {
            if (std::string(argv[1]) == "--quiet") {
                display_images = false;
            }
        }
    }
    if (display_images) {
        cv::namedWindow("raw", cv::WINDOW_AUTOSIZE);
        //cv::namedWindow("target", cv::WINDOW_AUTOSIZE);
        //cv::namedWindow("grad", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("map", cv::WINDOW_AUTOSIZE);
    }

    //double prev_pose[5];
    //vector<line_t> prev_points;
    vector<vptr> prev_lines;
    vector<vptr> saved_lines;

    // map_w x map_w 3-channel 8-bit image, initial value 255
    Mat map_img(map_w, map_w, CV_8UC3, {255, 255, 255});
    Mat observe_mask(map_w, map_w, CV_32FC1);
    Mat disp_map(map_w, map_w, CV_8UC3);
    Mat circle_mask(map_w, map_w, CV_32FC1);
    Mat new_lines(map_w, map_w, CV_8UC3);

    motion_dtype estimated_rot_err;

    motion_dtype* pose = NULL;
    double prev_head_tmp;
    std::deque<vptr> pose_queue;
    const size_t VIDEO_DELAY = 0;

    Mat disp(480, 640, CV_8UC3);
    Mat undistort;
    Mat deriv;

    int keyframe_count = 0;
    const int KEYFRAME_MIN = 5;
    size_t frame = 0;
    for (;;++frame) {
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
        if (camera_mode) {
            bool res = camera.read(disp);
        }
        else {
            size_t nbytes = (disp.dataend - disp.datastart) * sizeof(uchar);
            memcpy(disp.data, image_buffer, nbytes);
        }
        //cv::cvtColor(disp, disp, cv::COLOR_BGR2GRAY);
        cv::undistort(disp, undistort, camera_mat, distortion);

        while (!pose_gotten) {};
        ptree pt;
        read_json(pose_info, pt);
        vptr new_pose =  (vptr) malloc(5*sizeof(motion_dtype));
        new_pose[0] = pt.get<double>("x");
        new_pose[1] = pt.get<double>("y");
        new_pose[2] = pt.get<double>("heading");
        new_pose[3] = pt.get<double>("v");
        new_pose[4] = pt.get<double>("w");

        if (pose == NULL) {
            pose = new_pose;
            motion_dtype scratch[2];
            scratch[0] = pose[0];
            scratch[1] = pose[1];
            map_scale(scratch, scratch);
            cv::circle(map_img, _Point(scratch), map_scaling * 0.5, {0, 0, 0}, -1);
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
            //printf("raw pose: %f %f %f\n", new_pose[0], new_pose[1], new_pose[2]);
            //printf("heading change %f\n", new_pose[2] - prev_head_tmp);
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
        
        observe_mask = 0.0;
        disp_map = map_img.clone();
        new_lines = (char)0;
        vector<vptr> proj_lines;
        for (auto line : lines) {
            vptr proj_line = fast_alloc_vec(5);
            motion_dtype scratch[3];
            transform_point(scratch, line_first(line));
            __se3_apply(line_first(proj_line), camera_pose, scratch);
            transform_point(scratch, line_second(line));
            __se3_apply(line_second(proj_line), camera_pose, scratch);
            proj_line[4] = 0;
            proj_lines.push_back(proj_line);
        }
        proj_lines = dbscan_filter_lines(proj_lines, prev_lines, 0.25);
        ++keyframe_count;
        motion_dtype pose_x;
        motion_dtype pose_y;
        motion_dtype heading;
        if (keyframe_count >= KEYFRAME_MIN) {
            vector<line_t> saved_subset;
            pose_x = pose[0];
            pose_y = pose[1];
            heading = normalize_angle(pose[2]);
            for (auto line_score : saved_lines) {
                motion_dtype line_rel[4];
                __vo_sub(line_first(line_rel), line_first(line_score), pose_x, 2);
                __vo_sub(line_second(line_rel), line_second(line_score), pose_y, 2);
                motion_dtype angle1 = normalize_angle(atan2(line_rel[1], line_rel[0]));
                motion_dtype angle2 = normalize_angle(atan2(line_rel[3], line_rel[2]));
                if (angle_distance(angle1, heading) < camera_info.fov_x
                        || angle_distance(angle2, heading) < camera_info.fov_x) {
                    saved_subset.push_back(line_score);
                }
                else if (!(angle_distance(angle1, heading) < camera_info.fov_x/2
                        || angle_distance(angle2, heading) < camera_info.fov_x/2)) {
                    // TODO hack to exclude not in view lines from mismatch penalty...
                    ++line_score[4];
                }
            }
            vector<line_t> match_subset;
            for (auto line_score : proj_lines) {
                if (line_score[4] > 2) {
                    match_subset.push_back(line_score);
                }
            }
            motion_dtype best_tup[3];
            motion_dtype best_loss = register_lines(best_tup, match_subset, saved_subset);
            motion_dtype t = best_tup[0];
            motion_dtype dx = best_tup[1];
            motion_dtype dy = best_tup[2];

            vector<line_t> transformed_lines;
            motion_dtype r00 = cos(t);
            motion_dtype r10 = sin(t);
            motion_dtype r01 = -sin(t);
            motion_dtype r11 = r00;
            for (auto to_move : match_subset) {
                line_t moved = fast_alloc_vec(4);
                moved[0] = r00*to_move[0] + r01*to_move[1] + dx;
                moved[1] = r10*to_move[0] + r11*to_move[1] + dy;
                moved[2] = r00*to_move[2] + r01*to_move[3] + dx;
                moved[3] = r10*to_move[2] + r11*to_move[3] + dy;
                transformed_lines.push_back(moved);
            }
            saved_lines = dbscan_filter_lines(transformed_lines, saved_lines, 0.25);
            for (auto line : saved_lines) {
                if (line[4] > 2) {
                    line[4] = 2;
                }
            }

            pose[0] += dx;
            pose[1] += dy;
            pose[2] += t;
            estimated_rot_err -= t;
            if (display_images)  {
                Mat tmp_map = disp_map.clone();
                vector<line_t> scaled_tmp;
                for (auto l : saved_lines) {
                    line_t scaled_line = fast_alloc_vec(4);
                    map_scale(line_first(scaled_line), line_first(l));
                    map_scale(line_second(scaled_line), line_second(l));
                }
                plot_lines(tmp_map, scaled_tmp, {0, 255, 0});
                draw_robot(tmp_map, pose);
                cv::imshow("map", tmp_map);
            }

            proj_lines = vector<vptr>();
            for (auto line_score : saved_lines) {
                motion_dtype line_rel[4];
                __vo_sub(line_first(line_rel), line_first(line_score), pose_x, 2);
                __vo_sub(line_second(line_rel), line_second(line_score), pose_y, 2);
                motion_dtype angle1 = normalize_angle(atan2(line_rel[1], line_rel[0]));
                motion_dtype angle2 = normalize_angle(atan2(line_rel[3], line_rel[2]));
                if (angle_distance(angle1, heading) < camera_info.fov_x
                        || angle_distance(angle2, heading) < camera_info.fov_x) {
                    proj_lines.push_back(line_score);
                }
            }
            keyframe_count = 0;
        }
        
        // if prev points not none:
        //     plot the tracking points
        pose_x = pose[0];
        pose_y = pose[1];
        heading = normalize_angle(pose[2]);
        motion_dtype max_angle = heading + camera_info.fov_x/2;
        motion_dtype min_angle = heading - camera_info.fov_x/2;
        motion_dtype pose_center[2] = { pose_x, pose_y };
        motion_dtype pose_px[2];
        map_scale(pose_px, pose_center);
        const motion_dtype max_depth = 1;
        const motion_dtype scale = map_scaling * max_depth;
        circle_mask = 0.0;
        cv::circle(circle_mask, _pose_px, scale*3, 0.25, -1);
        cv::circle(circle_mask, _pose_px, scale*2, 0.5, -1);
        cv::circle(circle_mask, _pose_px, scale*1, 1, -1);
        cv::circle(circle_mask, _pose_px, scale*0.28, 0, -1);

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
        map_scale(min_pt, min_pt);
        cv::line(disp_map, center, _Point(min_pt), {255, 0, 0});

        Mat tmp = observe_mask * 0.10;
        cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);
        Mat tmp3;
        cv::multiply(tmp, map_img, tmp3, 1, CV_8UC3);
        cv::subtract(map_img, tmp3, tmp, Mat(), CV_8UC3);
        //cv::imshow("target", tmp3);
        Mat tmp2 = observe_mask.clone();
        cv::cvtColor(tmp2, tmp2, cv::COLOR_GRAY2BGR);
        cv::multiply(tmp2, new_lines, tmp3, 1, CV_8UC3);
        cv::add(tmp, tmp3, map_img, Mat(), CV_8UC3);

        prev_lines = proj_lines;

        if (display_images) {
            cv::imshow("raw", disp);
            //cv::imshow("grad", deriv);
            cv::imshow("map", disp_map);
            int key = cv::waitKeyEx(1);
            if (key == 27 || key == 'q') {
                break;
            }
        }

        size_t nbytes = (map_img.dataend - map_img.datastart) * sizeof(uchar);
        memcpy(map_buffer, map_img.data, nbytes);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "SPF: " << elapsed_seconds.count() <<  std::endl;
//        std::stringstream filename("calibration/cpp_log/capture_", std::ios_base::app | std::ios_base::out);
//        filename << std::setfill('0') << std::setw(6) << frame << ".png";
//        cv::imwrite(filename.str(), disp);
//
//        std::stringstream pose_fname("calibration/cpp_log/pose_", std::ios_base::app | std::ios_base::out);
//        pose_fname << std::setfill('0') << std::setw(6) << frame << ".json";
//        std::ofstream pose_file(pose_fname.str());
//        pose_file << pose_info.str();
//        pose_file.close();
    }
    return 0;
}
