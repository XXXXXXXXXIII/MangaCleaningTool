#include "mct.h"
#include "bubble.h"
#include "frame.h"
#include "text.h"
#include "io.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <tesseract/baseapi.h>

#include <iostream>
#include <map>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace mct;
namespace ts = tesseract;

int main()
{
    map<string, Mat> images = loadImages(selectImageFromDialog());
    Ptr<BoostFrameClassifier> frame_boost = new BoostFrameClassifier();
    Ptr<TesseractTextDetector> tess = new TesseractTextDetector();

    //tess.SetVariable("save_best_choices", "T");
    for (auto& img : images)
    {
        Mat img_gray = toGrayScale(img.second);

        

        auto timer = startTimer();
        vector<Frame> frames = extractFrame(img_gray);
        frame_boost->classifyFrame(frames);
        cleanFrame(img_gray, frames);
        cout << "Frame: " << stopTimer(timer) << endl;

        timer = startTimer();
        vector<Bubble> bubbles = findBubble(img_gray);
        cout << "Bubble: " << stopTimer(timer) << endl;

        timer = startTimer();
        
        tess->detextBubbleTextLine(img_gray, bubbles);
        
        cout << "Text: " << stopTimer(timer) << endl;



        cleanBubble(img_gray, bubbles);
        //showImage(img_gray);
    }

    waitKey(0);
    return 0;
}
