
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

	for (auto img : images)
	{
		Mat img_gray = toGrayScale(img.second);

		if (true) // Extract frame
		{
			vector<Frame> frames = extractFrame(img_gray);
			manualFrameSorter(img_gray, frames);
			saveFrameProperty(frames);
		}

		//Mat bubble_mask;
		//findBubbleCandidate(img.second, bubble_mask, false);
		//vector<Rect> text = findTextCandidate(img.second);
		//vector<BubbleProperty> bprop = extractBubbleProperty(bubble_mask, text);
		//manualSorter(img.second, bprop);
		//saveBubbleProperty(bprop);
		//saveCutouts(img.second, bprop, 100, 100);
	}

	cout << "All image processed!" << endl;
	waitKey(0);
}