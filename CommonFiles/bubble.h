#pragma once

#include "bubble_cascade_xml.h"
#include "bubble_boost_yaml.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

namespace mct
{
    struct Bubble
    {
        bool is_bubble = true;
        cv::Size page;
        cv::Rect box;
        std::vector<cv::Point> contour;
        std::vector<cv::Rect> text;

        // relative width
        // relative height
        // convexity
        // fill percentage (?)
        // text coverage
        // relative text location
        // hu moments
        std::vector<float> toInputData()
        {
            double contour_area = cv::contourArea(contour);
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);
            cv::Moments m = cv::moments(contour, true);
            double hu_moments[7];
            cv::HuMoments(m, hu_moments);
            for (int i = 0; i < 7; i++)
            {
                hu_moments[i] = -1 * std::copysign(1., hu_moments[i]) * log10(abs(hu_moments[i]));
            }

            return std::vector<float>{
                    (float)box.width / page.width,
                    (float)box.height/ page.height,
                    (float)contour_area / (float)cv::contourArea(hull),
                    (float)contour_area / box.area(),
                    //text_coverage,
                    //text_offset,
                    (float)hu_moments[0],
                    (float)hu_moments[1],
                    (float)hu_moments[2],
                    (float)hu_moments[3],
                    (float)hu_moments[4],
                    (float)hu_moments[5],
                    (float)hu_moments[6]};
        }

        std::string toCSVData()
        {
            std::vector<float> data = toInputData();
            return std::to_string(is_bubble) + ", "
                + std::to_string(data[0]) + ", "
                + std::to_string(data[1]) + ", "
                + std::to_string(data[2]) + ", "
                + std::to_string(data[3]) + ", "
                + std::to_string(data[4]) + ", "
                + std::to_string(data[5]) + ", "
                + std::to_string(data[6]) + ", "
                + std::to_string(data[7]) + ", "
                + std::to_string(data[8]) + ", "
                + std::to_string(data[9]) + ", "
                + std::to_string(data[10]) + ", "
                + std::to_string(data[11]) + ", "
                + std::to_string(data[12]) + ", "
                + std::to_string(data[13]) + ", "
                + std::to_string(data[14]);
        }
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