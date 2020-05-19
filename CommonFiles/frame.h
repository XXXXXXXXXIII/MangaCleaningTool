#pragma once

#ifndef __MCT_FRAME_H__
#define __MCT_FRAME_H__

#include "frame_boost_yaml.h"

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

namespace mct
{
	struct Frame
	{
		cv::Size page;
		cv::Rect box;
		std::vector<cv::Point> contour;
		bool is_frame = true;

		/*
			- is_frame
			- relative location x
			- relative location y
			- relative width
			- relative height
			- relative area
		*/
		std::string toCSVData()
		{
			return std::to_string(is_frame) + ", "
				+ std::to_string((float)(box.tl().x + box.width / 2) / page.width) + ", "
				+ std::to_string((float)(box.tl().y + box.height / 2) / page.height) + ", "
				//+ std::to_string((float)box.br().x / page.width) + ", "
				//+ std::to_string((float)box.br().y / page.height) + ", "
				+ std::to_string((float)box.width / page.width) + ", "
				+ std::to_string((float)box.height / page.height) + ", "
				+ std::to_string((float)box.area() / page.area());/* + ", "
				+ std::to_string((float)box.tl().x / page.width > 0.8 
					|| (float)box.tl().y / page.height > 0.8
					|| (float)box.br().x / page.width < 0.2
					|| (float)box.br().y / page.height < 0.2);*/
		}

		std::vector<float> toInputData()
		{
			return std::vector<float>{
					(float)(box.tl().x + box.width / 2) / page.width,
					(float)(box.tl().y + box.height / 2) / page.height,
					//(float)box.br().x / page.width,
					//(float)box.br().y / page.height,
					(float)box.width / page.width,
					(float)box.height / page.height,
					(float)box.area() / page.area()};/* ,
				(float)((float)box.tl().x / page.width > 0.8
					|| (float)box.tl().y / page.height > 0.8
					|| (float)box.br().x / page.width < 0.2
					|| (float)box.br().y / page.height < 0.2)};*/
		}
	};

	std::vector<Frame> extractFrame(const cv::Mat& image);
	void cleanFrame(cv::Mat& img, const std::vector<Frame>& frames, uchar back_color = 255);
	cv::Mat createFrameMask(const cv::Size& size, const std::vector<Frame>& frames);
	cv::Mat erasePatternFrame(const cv::Mat& image);

	class BoostFrameClassifier
	{
	private:
		cv::Ptr<cv::ml::Boost> frameClassifier;

	public:
		BoostFrameClassifier(std::string str_model = frame_boost_model);
		void classifyFrame(Frame& frame);
		void classifyFrame(std::vector<Frame>& frames);
	};
};

#endif