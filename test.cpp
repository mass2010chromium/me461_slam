#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>

int main()
{
    cv::Mat image1;
    image1 = cv::imread("known_patch0.pgm", 1);
    cv::Mat image2;
    image2 = cv::imread("known_patch1.pgm", 1);
    cv::Mat image3;
    
    cv::cvtColor(image1, image3, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(image3, points, 15, 0.01, 0);
    
    for (auto point : points) {
        cv::circle(image1, point, 2, {255, 0, 0});
    }
 
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image1);
    cv::waitKey(0);
    cv::imshow("Display Image", image2);
    cv::waitKey(0);
    return 0;
}
