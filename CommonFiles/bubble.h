#pragma once

#ifndef __MCT_BUBBLE_H__
#define __MCT_BUBBLE_H__

#include "text.h"

#include "bubble_cascade_xml.h"
#include "bubble_boost_yaml.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

namespace mct
{
    struct Text;
    struct Bubble
    {
        bool is_bubble = true;
        cv::Size page;
        cv::Rect box;
        std::vector<cv::Point> contour;
        std::vector<Text> text;

        std::string toCSVData();
        std::vector<float> toInputData();
    };

    std::vector<Bubble> findBubble(const cv::Mat& image);
    void cleanBubble(cv::Mat& img, const std::vector<Bubble>& bubbles, uchar bubble_color = 255, const cv::Point& offset = cv::Point(0, 0));
    void extractBubble(cv::Mat& img, const std::vector<Bubble>& bubbles, const uchar mask_color = 255, const cv::Point& offset = cv::Point(0,0));
    cv::Mat createBubbleMask(const cv::Size& size, const std::vector<Bubble>& bubbles, const uchar back_color = 0, const uchar bbl_color = 255, const cv::Point& offset = cv::Point(0, 0));

    class BoostBubbleClassifier
    {
    private: 
        cv::Ptr<cv::ml::Boost> bubbleClassifier;
    public:
        BoostBubbleClassifier(std::string str_model = bubble_boost_model);
        void classifyBubble(Bubble& bubble);
        void classifyBubble(std::vector<Bubble>& bubbles);
    };

    class CascadeBubbleDetector
    {
    private:
        cv::CascadeClassifier bubbleDetector;

    public:
        CascadeBubbleDetector(std::string cascade_file = bubble_cascade_model);
        void detectBubble(const cv::Mat& image, std::vector<Bubble>& bubbles);
    };
}

#endif