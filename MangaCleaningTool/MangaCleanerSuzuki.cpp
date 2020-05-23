#include "mct.h"
#include "bubble.h"
#include "frame.h"
#include "text.h"
#include "io.h"
#include "psd.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/features2d.hpp>
#include <tesseract/baseapi.h>

#include "psd_sdk/Psd.h"
#include "psd_sdk/PsdPlatform.h"
#include "psd_sdk/PsdExport.h"

#include <iostream>
#include <map>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace mct;
namespace ts = tesseract;

int main()
{
    map<wstring, Mat> images = loadImages(selectImageFromDialog());
    Ptr<BoostFrameClassifier> frame_boost = new BoostFrameClassifier();
    Ptr<BoostBubbleClassifier> bubble_boost = new BoostBubbleClassifier();
    Ptr<TesseractTextDetector> tess = new TesseractTextDetector(); // Inefficient
    Ptr<MSERTextDetector> text_mser = new MSERTextDetector(); // Need filtering
    //Ptr<MSERBubbleDetector> bubble_mser = new MSERBubbleDetector();
    //Ptr<EASTTextDetector> east = new EASTTextDetector(); // Poor accuracy

    //tess.SetVariable("save_best_choices", "T");
    for (auto& img : images)
    {
        Mat img_gray = toGrayScale(img.second);      

        printf("Width: %d   Height: %d   \n", img.second.cols, img.second.rows);

        cleanImage(img_gray);

        auto timer = startTimer();
        vector<Frame> frames = extractFrame(img_gray);
        frame_boost->classifyFrame(frames);
        straightenImage(img_gray, frames);
        //cleanFrame(img_gray, frames); //TODO: Not very working
        cout << "Frame: " << stopTimer(timer) << endl;

        timer = startTimer();
        vector<Bubble> bubbles = findBubbleCandidate(img_gray);
        //bubble_mser->detectBubble(img_gray);
        cout << "Bubble: " << stopTimer(timer) << endl;
        
        timer = startTimer();
        //tess->detextBubbleTextLine(img_gray, bubbles);
        //east->detectBubbleText(img.second, bubbles);        
        text_mser->detectBubbleTextLine(img_gray, bubbles);
        cout << "Text: " << stopTimer(timer) << endl;

        timer = startTimer();
        bubble_boost->classifyBubble(bubbles);
        cout << "Bubble Classifier: " << stopTimer(timer) << endl;

        cleanBubble(img_gray, bubbles);
        //showImage(img_gray, "image", 0.5);


        Mat img_mask(img.second.size(), CV_8UC2);
        vector<Mat> ch_mask;
        split(img_mask, ch_mask);
        bitwise_or(createBubbleMask(img.second.size(), bubbles, 0, 255),
            createFrameMask(img.second.size(), frames, 255, 0), ch_mask[0]);
        bitwise_or(ch_mask[0], ch_mask[1], ch_mask[1]);
        merge(ch_mask, img_mask);

        saveImage(img_gray, img.first);

        //psd::ExportDocument* psd_doc = createPsdDocument(img.second.size(), CV_8UC3);
        //addPsdLayer(psd_doc, img.second, "Original");
        //addPsdLayer(psd_doc, img_gray, "Level");
        //addPsdLayer(psd_doc, img_mask, "Mask");
        //savePsdDocument(psd_doc, img.first);
        //writePSD(toGrayScale(img.second), img_mask, img.first);
    }

    cv::waitKey(0);
    return 0;
}
