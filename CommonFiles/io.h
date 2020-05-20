#pragma once

#ifndef __MCT_IO_H__
#define __MCT_IO_H__

#include <mct.h>

#include <opencv2/core.hpp>
#include <map>

namespace mct
{
    const std::string bubble_prop_file = "bubble_property.csv";
    const std::string frame_prop_file = "frame_property.csv";

    std::vector<std::wstring> selectImageFromDialog(void);
    std::map<std::wstring, cv::Mat> loadImages(const std::vector<std::wstring>& filenames);
    int writePSD(const cv::Mat& img_original, const cv::Mat& img_mask, std::wstring filename);

    void saveFrameProperty(std::vector<Frame>& frames);
    void saveBubbleProperty(std::vector<Bubble>& bubbles);

    void saveCutouts(const cv::Mat& image, std::vector<Bubble>& bubbles, int width = 50, int height = 50);
    void saveCutouts(const cv::Mat& image, std::vector<cv::Rect>& rect, std::vector<bool>& tag, int width = 50, int height = 50);
};

#endif