#pragma once

#include "bubble.h"

#include <opencv2/core.hpp>
#include <tesseract/baseapi.h>

namespace mct
{
	struct Text
	{
		cv::Rect box;
		int fill_area = -1;
		float avg_width = -1;
	};

	cv::Mat strokeWidthTransform(const cv::Mat& img_edge, float angle = 3, bool dark_text = true);
	std::vector<cv::Point> connectedNeighbor(const cv::Mat& img, const cv::Point& p, float loFillRatio, float hiFillRatio, int scan_radius);
	int connectSWTComponents(const cv::Mat& img_swt, cv::Mat& img_ccl, float loFillDiff = 5, float hiFillDiff = 5, int scan_radius = 1);
	std::vector<cv::Rect> textCandidateFilter(const cv::Mat& img_ccl);
	std::vector<Text> findTextCandidate(const cv::Mat& image);

	class TesseractTextDetector
	{
	private:
		tesseract::TessBaseAPI tess;

	public:
		TesseractTextDetector()
		{
			tess.Init("C:\\Users\\timxi\\source\\repos\\MangaCleaningTool\\MangaCleaningTool\\", "jpn+jpn_vert", tesseract::OEM_LSTM_ONLY);
		}

		void detextBubbleTextLine(const cv::Mat& image, const std::vector<Bubble>& bubbles);
		//void detextBubbleTextChar();
	};
}