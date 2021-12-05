#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>

#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main()
{
    std::ifstream t("intrinsics.json");
    std::stringstream buffer;
    buffer << t.rdbuf();
    auto intrinsics = json::parse(buffer.str());
    auto json_mat = intrinsics["matrix"];
    auto json_distort = intrinsics["distortion"];
    cv::Mat camera_mat = cv::Mat::zeros(3, 3, CV_64F);
    std::vector<double> distortion;

    for (const double& d : json_distort[0]) {
        distortion.push_back(d);
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            camera_mat.at<double>(i, j) = json_mat[i][j];
        }
    }
    std::cout << camera_mat << std::endl;

    cv::VideoCapture camera(0);
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::Mat image;
    cv::Mat undistort;

    const int N = 20;
    while (true) {
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < N; ++i) {
            bool res = camera.read(image);
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
            cv::undistort(image, undistort, camera_mat, distortion);
        }
        //cv::imshow("Display Image", image);
        //cv::waitKey(1);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "FPS: " << N / elapsed_seconds.count() << " " << image.size <<  std::endl;
    }
    return 0;
}
