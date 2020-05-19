#pragma once

#ifndef __MCT_H__
#define __MCT_H__

#include <frame.h>
#include <bubble.h>
#include <text.h>
#include <io.h>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

namespace mct
{
    void showImage(const cv::Mat& image, std::string name = "image", float size = 0.7);
    void showImageWithMouse(const cv::Mat& image, float size = 0.7);

    int tagImage(const cv::Mat& display_image);
    std::vector<bool> manualSorter(const cv::Mat& image, std::vector<cv::Rect>& rect, const cv::Size& display_size = cv::Size(100, 100));
    void manualFrameSorter(const cv::Mat& image, std::vector<Frame>& frames);
    void manualBubbleSorter(const cv::Mat& image, std::vector<Bubble>& bubbles);

    int getopt(int argc, char** argv, char* optarg);

    cv::Mat toGrayScale(const cv::Mat& img_in);
    cv::Scalar bgColor(const cv::Mat& img, int borderWidth = 10);

    void drawDottedRect(cv::Mat& img, cv::Rect r, cv::Scalar dot_color = cv::Scalar(255), int dot_length = 2, int dot_height = 4, int dot_spacing = 4);
    double pointDist(cv::Point a, cv::Point b);
    double pointDist(cv::Point2f a, cv::Point2f b);
    int compareRect(cv::Rect r1, cv::Rect r2);
    int compareContour(std::vector<cv::Point>& c1, std::vector<cv::Point>& c2);
    int minEdgeDist(cv::Rect r, cv::Point p);

    std::chrono::steady_clock::time_point startTimer(void);
    double stopTimer(std::chrono::steady_clock::time_point startTime);
};

#endif