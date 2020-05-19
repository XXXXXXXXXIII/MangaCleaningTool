#include "bubble.h"

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

namespace mct
{
	BoostBubbleClassifier::BoostBubbleClassifier(std::string str_model)
	{
		this->bubbleClassifier = Boost::loadFromString<Boost>(cv::String(str_model));
	}

	void BoostBubbleClassifier::classifyBubble(Bubble& bubble)
	{
		Mat result;
		this->bubbleClassifier->predict(bubble.toInputData(), result);
		bubble.is_bubble = result.at<float>(0, 0) > 0;
	}

	void BoostBubbleClassifier::classifyBubble(std::vector<Bubble>& bubbles)
	{
		for (int i = 0; i < bubbles.size(); i++)
		{
			Mat result;
			this->bubbleClassifier->predict(bubbles[i].toInputData(), result);
			bubbles[i].is_bubble = result.at<float>(0, 0) > 0;
		}
	}
};