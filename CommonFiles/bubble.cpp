#include "bubble.h"
#include "mct.h"
#include "text.h"

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

namespace mct
{
    std::vector<float> Bubble::toInputData()
    {
        double contour_area = cv::contourArea(contour);
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        cv::Moments m = cv::moments(contour, true);
        double hu_moments[7];
        cv::HuMoments(m, hu_moments);
        for (int i = 0; i < 7; i++)
        {
            hu_moments[i] = -1 * std::copysign(1., hu_moments[i]) * log10(abs(hu_moments[i]));
        }

        float text_box_area = 0;
        float text_stroke_area = 0;
        float text_stroke_width = 0;
        cv::Rect text_bound = Rect(0, 0, 0, 0);
        if (text.size() > 0)
        {
            for (auto& t : text)
            {
                text_box_area += t.box.area();
                text_stroke_area += t.fill_area;
                text_stroke_width += t.fill_area * t.avg_width;
                text_bound |= t.box;
            }
            if (text_stroke_area > 0) text_stroke_width /= text_stroke_area;
        }

        return std::vector<float>{
            (float)box.width / page.width,
                (float)box.height / page.height,
                (float)contour_area / (float)cv::contourArea(hull),
                (float)contour_area / box.area(),
                (float)text_box_area / (float)contour_area,
                (float)text_stroke_area / (float)contour_area,
                (float)(text_bound.tl().x + text_bound.width / 2) / box.width,
                (float)(text_bound.tl().y + text_bound.height / 2) / box.height,
                (float)text_bound.width / box.width,
                (float)text_bound.height / box.height,
                (float)hu_moments[0],
                (float)hu_moments[1],
                (float)hu_moments[2],
                (float)hu_moments[3]};
    }

    std::string Bubble::toCSVData()
    {
        std::vector<float> data = toInputData();
        return std::to_string(is_bubble) + ", "
            + std::to_string(data[0]) + ", "
            + std::to_string(data[1]) + ", "
            + std::to_string(data[2]) + ", "
            + std::to_string(data[3]) + ", "
            + std::to_string(data[4]) + ", "
            + std::to_string(data[5]) + ", "
            + std::to_string(data[6]) + ", "
            + std::to_string(data[7]) + ", "
            + std::to_string(data[8]) + ", "
            + std::to_string(data[9]) + ", "
            + std::to_string(data[10]) + ", "
            + std::to_string(data[11]) + ", "
            + std::to_string(data[12]) + ", "
            + std::to_string(data[13]) + ", "
            + std::to_string(data[14]) + ", "
            + std::to_string(data[15]) + ", "
            + std::to_string(data[16]) + ", "
            + std::to_string(data[17]);
    }

	vector<Bubble> findBubble(const Mat& image)
	{
        //for (float i = 0; i < 10; i+=1)
        //{
            //Ptr<MSER> mser = MSER::create(30, 
            //    image.size().area() / 3000, 
            //    image.size().area() / 10, 
            //    5);
            //vector<vector<Point>> pts;
            //vector<Rect> box;
            //Mat img = image.clone();
            //bilateralFilter(image, img, 5, 50, 50);
            //threshold(img, img, 240, 255, THRESH_BINARY);
            //mser->detectRegions(image, pts, box);
            //for (auto& b : box)
            //{
            //    rectangle(img, b, Scalar(180), 2);
            //}
            ////cout << i << endl;
            //showImage(img, "", 0.3);
        //}

        Mat img_bin;
        const uchar bubbleColor = 180, bubbleColorH = 181;
        bilateralFilter(image, img_bin, 5, 50, 50); // Remove noise on image //TODO: Find better simga value
        threshold(img_bin, img_bin, 240, 255, THRESH_BINARY); // threshold 240 as recommended in the paper

        // CCL Filter
        Mat img_ccl(image.size(), CV_32S);
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        //TODO: what if bubble have larger openings?
        morphologyEx(img_bin, img_bin, MORPH_OPEN, kernel, Point(-1, -1), 1);
        int nLabels = connectedComponents(img_bin, img_ccl, 4);
        vector<Rect> blobs(nLabels);
        vector<int> areas(nLabels);
        vector<Point> seedPoints(nLabels);

        for (int y = 0; y < img_ccl.rows; y++) {
            for (int x = 0; x < img_ccl.cols; x++) {
                int label = img_ccl.at<int>(y, x);
                areas[label]++;
                if (blobs[label].empty())
                {
                    seedPoints[label] = Point(x, y);
                    blobs[label] = Rect(x, y, 1, 1);
                }
                else
                {
                    if (blobs[label].x > x)
                        blobs[label].x = x;
                    else if (blobs[label].x + blobs[label].width < x)
                        blobs[label].width = x - blobs[label].x;

                    if (blobs[label].y > y)
                        blobs[label].y = y;
                    else if (blobs[label].y + blobs[label].height < y)
                        blobs[label].height = y - blobs[label].y;
                }
            }
        }

        // Basic filter
        vector<int> blob_candidates;
        for (int i = 0; i < blobs.size(); ++i)
        {
            if (blobs[i].width < 20 // TODO: Need a better guess here
                || blobs[i].height < 20
                || blobs[i].area() > areas[i] * 2
                || blobs[i].area() > image.size().area() / 2)
            {
                continue;
            }
            blob_candidates.push_back(i);
        }

        for (int i : blob_candidates)
        {
            Rect r = blobs[i];

            if (img_bin.at<uchar>(seedPoints[i]) != bubbleColor)
            {
                Rect fillBox;
                floodFill(img_bin, seedPoints[i], Scalar(bubbleColor), &fillBox);
                //rectangle(img_bin, blobs[i], Scalar(100), 2);
                //blob_elected.push_back(fillBox);
            }
        }

        inRange(img_bin, Scalar(bubbleColor), Scalar(bubbleColorH), img_bin);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        vector<Bubble> bubbles;
        findContours(img_bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        // HACK: Only select contour whose child does not have child (aka bubble with text), seem to work
        for (int i = 0; i < hierarchy.size(); i++)
        {
            Vec4i h = hierarchy[i];
            if (h[2] > 0 && hierarchy[h[2]][2] < 0)
            {
                Bubble bubble = { true, image.size(), boundingRect(contours[i]), contours[i] };
                bubbles.push_back(bubble);
                //drawContours(bubble_mask, contours, i, Scalar(128), FILLED);
            }
        }

        return bubbles;
	}

    void cleanBubble(Mat& img, const vector<Bubble>& bubbles, uchar bubble_color, const Point& offset)
    {
        Mat bubble_mask = createBubbleMask(img.size(), bubbles, 0, 255, offset);
        //Mat bubble_mask_inv;
        //bitwise_not(bubble_mask, bubble_mask_inv);
        //bitwise_and(img, bubble_mask, img);
        bitwise_and(img, Scalar(0), img, bubble_mask);
        bitwise_or(img, Scalar(bubble_color), img, bubble_mask);
    }

    void extractBubble(Mat& img, const vector<Bubble>& bubbles, const uchar mask_color, const Point& offset)
    {
        Mat mask = createBubbleMask(img.size(), bubbles, 255, 0, offset);
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(mask, mask, MORPH_DILATE, kernel, Point(-1, -1), 1);
        bitwise_and(img, Scalar(0), img, mask);
        bitwise_or(img, Scalar(mask_color), img, mask);
    }

    Mat createBubbleMask(const Size& size, const vector<Bubble>& bubbles, const uchar back_color, const uchar bbl_color, const Point& offset)
    {
        Mat mask(size, CV_8UC1, Scalar(back_color));
        for (const Bubble& b : bubbles)
        {
            if (!b.is_bubble) continue;
            drawContours(mask, vector<vector<Point>>{ b.contour }, -1, Scalar(bbl_color), FILLED, 8, noArray(), INT32_MAX, offset);
        }
        return mask;
    }
}