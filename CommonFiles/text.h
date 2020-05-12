#pragma once

#include <opencv2/core.hpp>

namespace mct
{
	cv::Mat strokeWidthTransform(const cv::Mat& img_edge, float angle = 3, bool dark_text = true);
	std::vector<cv::Point> connectedNeighbor(const cv::Mat& img, const cv::Point& p, float loFillRatio, float hiFillRatio, int scan_radius);
	cv::Mat connectSWTComponents(const cv::Mat& img_swt, float loFillDiff = 5, float hiFillDiff = 5, int scan_radius = 1);
	std::vector<cv::Rect> textCandidateFilter(const cv::Mat& img_ccl);
	std::vector<cv::Rect> findTextCandidate(const cv::Mat& image);
}