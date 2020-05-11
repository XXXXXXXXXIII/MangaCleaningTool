#pragma once

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

		std::string toCSVData()
		{
			return std::to_string(is_frame) + ", "
				+ std::to_string((float)box.tl().x / page.width) + ", "
				+ std::to_string((float)box.tl().y / page.height) + ", "
				+ std::to_string((float)box.br().x / page.width) + ", "
				+ std::to_string((float)box.br().y / page.height) + ", "
				+ std::to_string((float)box.width / page.width) + ", "
				+ std::to_string((float)box.height / page.height) + ", "
				+ std::to_string((float)box.area() / page.area());
		}

		std::vector<float> toInputData()
		{
			return std::vector<float>{
				(float)box.tl().x / page.width,
				(float)box.tl().y / page.height,
				(float)box.br().x / page.width,
				(float)box.br().y / page.height,
				(float)box.width / page.width,
				(float)box.height / page.height,
				(float)box.area() / page.area()};
		}
	};

	std::vector<Frame> extractFrame(const cv::Mat& image);
	void cleanFrame(cv::Mat& img, const std::vector<Frame>& frames, uchar back_color = 255);
	cv::Mat createFrameMask(const cv::Size& size, const std::vector<Frame>& frames);

	class BoostFrameClassifier
	{
	private:
		cv::Ptr<cv::ml::Boost> frameClassifier;

	public:
		BoostFrameClassifier(std::string file = "frame_boost.yaml");
		void classifyFrame(Frame& f);
		void classifyFrame(std::vector<Frame>& f);
	};
};