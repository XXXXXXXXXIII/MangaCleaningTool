#include "bubble.h"
#include "mct.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

namespace mct
{
	//MSERBubbleDetector::MSERBubbleDetector()
	//{
	//	mser = MSER::create(30, 1, 100000, 5);
	//}

 //   //TODO: maybe faster than CCL?
	//vector<Bubble> MSERBubbleDetector::detectBubble(const Mat& image)
	//{
	//	mser->setMinArea(image.size().area() / 3000);
	//	mser->setMaxArea(image.size().area() / 10);

 //       vector<vector<Point>> pts;
 //       vector<Rect> box, box_all;
 //       Mat img_clone = image.clone();
 //       Mat img_bin;
 //       bilateralFilter(image, img_bin, 5, 50, 50);
 //       threshold(img_bin, img_bin, 240, 255, THRESH_BINARY);

 //       for (int i = 3; i <= 9; i += 2)
 //       {
 //           Mat temp = img_bin.clone();
 //           Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(i, i));
 //           morphologyEx(temp, temp, MORPH_ERODE, kernel);
 //           mser->detectRegions(temp, pts, box);
 //           for (auto& b : box)
 //           {
 //               box_all.push_back(b);
 //           }
 //       }


 //       for (auto& b : box_all)
 //       {
 //           rectangle(img_clone, b, Scalar(180), 2);
 //       }
 //       //cout << i << endl;
 //       showImage(img_clone);

 //       return vector<Bubble>();
	//}
};