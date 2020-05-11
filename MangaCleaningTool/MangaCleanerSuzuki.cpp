#include "mct.h"
#include "bubble.h"
#include "frame.h"
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
    Ptr<Boost> frame_boost = Boost::load("frame_boost.yaml");

    for (auto img : images)
    {
        Mat img_gray = toGrayScale(img.second);
        vector<Frame> frames = extractFrame(img_gray);

        for (int i = 0; i < frames.size(); i++)
        {
            Mat result;
            frame_boost->predict(frames[i].toInputData(), result);
            frames[i].is_frame = result.at<float>(0, 0) > 0;
        }

        cleanFrame(img_gray, frames);
        showImage(img_gray);
    }

    waitKey(0);
    return 0;
}
