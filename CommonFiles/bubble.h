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
#include <opencv2/features2d.hpp>
#include <opencv2/dnn.hpp>

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
        uchar color;

        std::string toCSVData();
        std::vector<float> toInputData();
    };

    std::vector<Bubble> findBubbleCandidate(const cv::Mat& image);
    void cleanBubble(cv::Mat& img, const std::vector<Bubble>& bubbles, uchar bubble_color = 255, const cv::Point& offset = cv::Point(0, 0));
    void extractBubble(cv::Mat& img, const std::vector<Bubble>& bubbles, const uchar mask_color = 255, const cv::Point& offset = cv::Point(0,0));
    cv::Mat createBubbleMask(const cv::Size& size, const std::vector<Bubble>& bubbles, const uchar back_color = 0, const uchar bbl_color = 255, const cv::Point& offset = cv::Point(0, 0));

    //class MSERBubbleDetector
    //{
    //private:
    //    cv::Ptr<cv::MSER> mser;
    //public:
    //    MSERBubbleDetector();
    //    std::vector<Bubble> detectBubble(const cv::Mat& image);
    //};

    class BoostBubbleClassifier
    {
    private: 
        cv::Ptr<cv::ml::Boost> bubbleClassifier;
    public:
        BoostBubbleClassifier(std::string str_model = bubble_boost_model);
        void classifyBubble(Bubble& bubble);
        void classifyBubble(std::vector<Bubble>& bubbles);
    };

    class TFBubbleClassifier
    {
    private:
        cv::dnn::Net model;

    public:
        TFBubbleClassifier(std::string path = "C:\\Users\\timxi\\source\\repos\\MangaCleaningTool\\CommonFiles\\bubble.pb");
        void classifyBubble(const cv::Mat& img, Bubble& bubble);
        void classifyBubble(const cv::Mat& img, std::vector<Bubble>& bubbles);
    };

    //class CascadeBubbleDetector
    //{
    //private:
    //    cv::CascadeClassifier bubbleDetector;

    //public:
    //    CascadeBubbleDetector(std::string cascade_file = bubble_cascade_model);
    //    void detectBubble(const cv::Mat& image, std::vector<Bubble>& bubbles);
    //};
}

#endif