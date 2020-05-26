#include "bubble.h"

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>

#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace cv::dnn;
namespace fs = std::filesystem;

namespace mct
{
	TFBubbleClassifier::TFBubbleClassifier(std::string path)
	{
		//writeTextGraph(path, "bubble.pbtxt");

		size_t size = fs::file_size(path);
		vector<uchar> buffer(size);
		ifstream ifs(path, ios::binary | ios::in);
		ifs.read(reinterpret_cast<char*>(&buffer[0]), size);
		ifs.close();
		this->model = readNet(path);
	}

	void TFBubbleClassifier::classifyBubble(const cv::Mat& img, Bubble& bubble)
	{
		Mat img_in;
		//vector<Mat> out;
		resize(Mat(img, bubble.box), img_in, Size(100, 100));
		model.setInput(img_in);
		Mat out = model.forward();

		cout << out.channels() << out.size() << endl;
	}

	void TFBubbleClassifier::classifyBubble(const cv::Mat& img, std::vector<Bubble>& bubbles)
	{
		for (auto& b : bubbles)
		{
			classifyBubble(img, b);
		}
	}
};