#pragma once

#ifndef __MCT_TEXT_H__
#define __MCT_TEXT_H__

#include "bubble.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/features2d.hpp>
//#include <tesseract/baseapi.h>

namespace mct
{
	struct Bubble;
	struct Text
	{
		cv::Rect box;
		int fill_area = -1;
		float avg_width = -1;
	};

	cv::Mat strokeWidthTransform(const cv::Mat& image, float angle = 3, bool dark_text = true);
	std::vector<cv::Point> connectedNeighbor(const cv::Mat& img, const cv::Point& p, float loFillRatio, float hiFillRatio, int scan_radius);
	int connectSWTComponents(const cv::Mat& img_swt, cv::Mat& img_ccl, float loFillDiff = 5, float hiFillDiff = 5, int scan_radius = 1);
	std::vector<cv::Rect> textCandidateFilter(const cv::Mat& img_ccl);
	std::vector<Text> findTextCandidate(const cv::Mat& image);

	//class TesseractTextDetector
	//{
	//private:
	//	tesseract::TessBaseAPI tess;

	//public:
	//	TesseractTextDetector()
	//	{
	//		tess.Init("C:\\Users\\timxi\\source\\repos\\MangaCleaningTool\\MangaCleaningTool\\", "jpn+jpn_vert", tesseract::OEM_LSTM_ONLY);
	//	}

	//	void detextBubbleTextLine(const cv::Mat& image, std::vector<Bubble>& bubbles);
	//	//void detextBubbleTextChar();
	//};

	class MSERTextDetector
	{
	private:
		enum class ORIENTATION
		{
			VERTICAL,
			HORIZONTAL
		};

		cv::Ptr<cv::MSER> mser;
		std::vector<Text> detectTextLine(const cv::Mat& image);
		std::vector<Text> joinTextLine(std::vector<Text>& text, ORIENTATION o = ORIENTATION::VERTICAL);
		std::vector<Text> detectTextChar(const cv::Mat& image);

	public:
		MSERTextDetector()
		{
			mser = cv::MSER::create(5, 1, 10000);
		}

		void detectBubbleTextLine(const cv::Mat& image, std::vector<Bubble>& bubbles);
	};

	class EASTTextDetector
	{
	private:
		cv::dnn::Net detector;
		void decodeBoundingBoxes(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh, std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences);
		void decodeText(const cv::Mat& scores, std::string& text);
		void fourPointsTransform(const cv::Mat& frame, cv::Point2f vertices[4], cv::Mat& result);

	public:
		EASTTextDetector()
		{
			detector = cv::dnn::readNet(R"()");
		}

		void detectBubbleText(const cv::Mat& image, const std::vector<Bubble>& bubbles);

	};
}

#endif