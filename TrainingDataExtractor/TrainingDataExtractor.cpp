
#include <mct.h>
#include <io.h>
#include <bubble.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <map>

using namespace std;
using namespace mct;
using namespace cv;

int main(int argc, char** argv)
{
	map<string, Mat> images = loadImages(selectImageFromDialog());
	Ptr<MSERTextDetector> mser = new MSERTextDetector(); // Need filtering
	Ptr<BoostFrameClassifier> frame_boost = new BoostFrameClassifier();

	for (auto img : images)
	{
		Mat img_gray = toGrayScale(img.second);

		if (false) // Extract frame
		{
			vector<Frame> frames = extractFrame(img_gray);
			manualFrameSorter(img_gray, frames);
			saveFrameProperty(frames);
		}

		if (true)
		{
			vector<Frame> frames = extractFrame(img_gray);
			frame_boost->classifyFrame(frames);
			cleanFrame(img_gray, frames);
			vector<Bubble> bubbles = findBubble(img_gray);
			mser->detectBubbleTextLine(img_gray, bubbles);
			manualBubbleSorter(img_gray, bubbles);
			saveBubbleProperty(bubbles);
			saveCutouts(img_gray, bubbles, 100, 100);
		}
	}

	cout << "All image processed!" << endl;
	waitKey(0);
}