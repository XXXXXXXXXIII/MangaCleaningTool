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
//#include <tesseract/baseapi.h>

#include "psd_sdk/Psd.h"
#include "psd_sdk/PsdPlatform.h"
#include "psd_sdk/PsdExport.h"

#include <iostream>
#include <map>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace mct;
//namespace ts = tesseract;

int main(int argc, char** args)
{
    bool doClean = false, doFrame = false, doShow = false;
    if (argc > 1)
    {
        if (string(args[1]).find('c') != string::npos)
        {
            doClean = true;
        }
        if (string(args[1]).find('f') != string::npos)
        {
            doFrame = true;
        }
        if (string(args[1]).find('s') != string::npos)
        {
            doShow = true;
        }
    }


    map<wstring, Mat> images = loadImages(selectImageFromDialog());
    Ptr<BoostFrameClassifier> frame_boost = new BoostFrameClassifier();
    Ptr<BoostBubbleClassifier> bubble_boost = new BoostBubbleClassifier();
    //Ptr<TesseractTextDetector> tess = new TesseractTextDetector(); // Inefficient
    Ptr<MSERTextDetector> text_mser = new MSERTextDetector(); // Need filtering
    //Ptr<TFBubbleClassifier> bubble_tf = new TFBubbleClassifier();
    //Ptr<MSERBubbleDetector> bubble_mser = new MSERBubbleDetector();
    //Ptr<EASTTextDetector> east = new EASTTextDetector(); // Poor accuracy

    //tess.SetVariable("save_best_choices", "T");
    for (auto& img : images)
    {
        Mat img_gray = toGrayScale(img.second);      

        printf("Width: %d   Height: %d   \n", img.second.cols, img.second.rows);


        auto timer = startTimer();
        if (doClean)
        {
            cleanImage(img_gray);
        }

        vector<Frame> frames;
        if (doFrame)
        {
            frames = extractFrame(img_gray);
            frame_boost->classifyFrame(frames);
        }
        //straightenImage(img_gray, frames);

        //frames = extractFrame(img_gray);
        //frame_boost->classifyFrame(frames);
        //Mat img_frameless = img_gray.clone();
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
        //bubble_tf->classifyBubble(img_gray, bubbles);
        bubble_boost->classifyBubble(bubbles);
        cout << "Bubble Classifier: " << stopTimer(timer) << endl;

        Mat img_out = img_gray.clone();
        cleanBubble(img_out, bubbles);
        if (doFrame)
        {
            cleanFrame(img_out, frames);
        }

        Mat img_mask(img.second.size(), CV_8UC2);
        vector<Mat> ch_mask;
        split(img_mask, ch_mask);
        ch_mask[0] = createBubbleMask(img.second.size(), bubbles, 0, 255);
        if (doFrame)
        {
            bitwise_or(createBubbleMask(img.second.size(), bubbles, 0, 255), createFrameMask(img.second.size(), frames, 255, 0), ch_mask[0]);
        }
        bitwise_or(ch_mask[0], ch_mask[1], ch_mask[1]);
        merge(ch_mask, img_mask);

        if (doShow)
        {
            showImage(img_out, "image");
            //saveImage(img_out, img.first);
        }
        else
        {
            psd::ExportDocument* psd_doc = createPsdDocument(img.second.size(), CV_8UC1);
            //addPsdLayer(psd_doc, img.second, "Original");
            addPsdLayer(psd_doc, img_gray, "Clean");
            addPsdLayer(psd_doc, img_mask, "Mask");
            savePsdDocument(psd_doc, img.first);
        }

    }

    cv::waitKey(0);
    return 0;
}
