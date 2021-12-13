#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>

#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
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

int main()
{
    std::ifstream t("intrinsics.json");
    std::stringstream buffer;
    buffer << t.rdbuf();
    ptree pt;
    read_json(buffer, pt);
    auto json_mat = pt.get_child("matrix");
    auto json_distort = pt.get_child("distortion");
    cv::Mat camera_mat = cv::Mat::zeros(3, 3, CV_64F);
    std::vector<double> distortion;

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

    cv::VideoCapture camera(0);
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::Mat image;
    cv::Mat undistort;

    const int N = 20;
    size_t frame = 0;
    for (;;) {
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < N; ++i) {
            HttpClient client("localhost:8080");
            volatile bool pose_gotten = false;
            std::string pose_info;
            client.request("GET", "/pose", "", [&pose_gotten, &pose_info](std::shared_ptr<HttpClient::Response> response, const SimpleWeb::error_code& ec) {
                if (!ec) {
                    std::stringstream s;
                    s << response->content.rdbuf();
                    pose_info = s.str();
                }
                else {
                    pose_info = R"({"x":0,"y":0,"heading":0,"v":0,"w":0})";
                    std::cout << "http request failed" << std::endl;
                }
                pose_gotten = true;
            });
            client.io_service->run();
            std::stringstream filename("cpp_log/capture_", std::ios_base::app | std::ios_base::out);
            bool res = camera.read(image);
            cv::undistort(image, undistort, camera_mat, distortion);
            filename << std::setfill('0') << std::setw(6) << frame << ".png";
            //cv::imwrite(filename.str(), image);

            std::stringstream pose_fname("cpp_log/pose_", std::ios_base::app | std::ios_base::out);
            pose_fname << std::setfill('0') << std::setw(6) << frame << ".json";
            std::ofstream pose_file(pose_fname.str());
            while (!pose_gotten) {};
            pose_file << pose_info;
            pose_file.close();
            ++frame;
        }
        cv::imshow("Display Image", undistort);
        cv::waitKey(1);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "FPS: " << N / elapsed_seconds.count() << " " << image.size <<  std::endl;
    }
    return 0;
}
