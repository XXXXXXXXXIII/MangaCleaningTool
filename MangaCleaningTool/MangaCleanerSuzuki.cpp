#include "mct.h"
#include "bubble.h"
#include "frame.h"
#include "text.h"
#include "io.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <map>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace mct;

int main()
{
    map<string, Mat> images = loadImages(selectImageFromDialog());
    Ptr<BoostFrameClassifier> frame_boost = new BoostFrameClassifier();

    for (auto& img : images)
    {
        Mat img_gray = toGrayScale(img.second);

        auto timer = startTimer();
        vector<Frame> frames = extractFrame(img_gray);
        frame_boost->classifyFrame(frames);
        cleanFrame(img_gray, frames);
        cout << "Frame: " << stopTimer(timer) << endl;

        timer = startTimer();
        vector<Rect> text = findTextCandidate(img_gray);
        cout << "Text: " << stopTimer(timer) << endl;

        timer = startTimer();
        vector<Bubble> bubbles = extractBubble(img_gray);
        cleanBubble(img_gray, bubbles);
        cout << "Bubble: " << stopTimer(timer) << endl;


        showImage(img_gray);
    }

    waitKey(0);
    return 0;
}
