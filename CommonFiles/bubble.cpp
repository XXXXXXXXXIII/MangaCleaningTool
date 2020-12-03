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

    /*
        Find connected components for the image. 
        Sweep through several erosion level to close any openings.
        //TODO: Support non-white bubble
    */
	vector<Bubble> findBubbleCandidate(const Mat& image)
	{
        vector<Bubble> bubbles;
        for (int s = 1; s <= 9; s += 2) //sweep through different erosion level to close bubble openings
        {
            Mat img_bin, img_ccl;
            const uchar bubbleColor = 180, bubbleColorH = 181;
            //bilateralFilter(image, img_bin, 5, 50, 50); // Remove noise on image
            threshold(image, img_bin, 240, 255, THRESH_BINARY); // threshold 240 as recommended in the paper

            // CCL Filter
            Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(s, s));
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
                    || blobs[i].area() > areas[i] * 4
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
                    //blob_elected.push_back(fillBox);
                }
            }

            inRange(img_bin, Scalar(bubbleColor), Scalar(bubbleColorH), img_bin);

            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(img_bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            //showImage(img_bin);
            // HACK: Only select contour whose child does not have child (aka bubble with text), seem to work
            for (int i = 0; i < hierarchy.size(); i++)
            {
                Vec4i h = hierarchy[i];
                if (h[2] > 0)
                {
                    int t = h[2];
                    double a = 0;
                    Mat mask = Mat::zeros(img_bin.size(), CV_8UC1);
                    while (t > 0)
                    {
                        if (hierarchy[t][2] > 0)
                        {
                            a = 0;
                            break;
                        }
                        a += contourArea(contours[t]);
                        t = hierarchy[t][0];
                    }
                    if (contourArea(contours[i]) * 0.02 > a) continue; // Remove contours with small fill percentage
                    drawContours(mask, contours, h[2], Scalar(255), FILLED);
                    if (mean(img_bin, mask)[0] == 255) continue; // Remove contours with non-text color content

                    Bubble bubble = { true, image.size(), boundingRect(contours[i]), contours[i] };
                    bubbles.push_back(bubble);

                }
            }
        }

        // Remove duplicates
        sort(bubbles.begin(), bubbles.end(), [](const Bubble& lhs, const Bubble& rhs)
            {
                return lhs.box.area() > rhs.box.area();
            });

        //Mat img_bin = Mat::zeros(image.size(), CV_8UC1);
        for (int i = 0; i < bubbles.size(); ++i)
        {
            if (!bubbles[i].is_bubble) continue;
            for (int j = i + 1; j < bubbles.size(); ++j)
            {
                if (0.9 * bubbles[i].box.area() > bubbles[j].box.area()) break;
                if (!bubbles[j].is_bubble) continue;
                if ((bubbles[i].box & bubbles[j].box) == bubbles[j].box
                    && contourArea(bubbles[i].contour) > contourArea(bubbles[j].contour))
                {
                    bubbles[j].is_bubble = false;
                }
            }

            //rectangle(img_bin, bubbles[i].box, Scalar(180), 1);
        }
        bubbles.erase(remove_if(bubbles.begin(), bubbles.end(),
                [](const Bubble& b) { return !b.is_bubble; }), bubbles.end());

        //showImage(image);
        //showImage(img_bin);

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
            //bool overlap = false;
            //for (const Bubble& b2 : bubbles)
            //{
            //    if (!b2.is_bubble || b.box == b2.box) continue;
            //    if (compareRect(b.box, b2.box) == 1)
            //    {
            //        overlap = true;
            //        break;
            //    }
            //}
            //if (!overlap)
            //{
            //}
            drawContours(mask, vector<vector<Point>>{ b.contour }, -1, Scalar(bbl_color), FILLED, 8, noArray(), INT32_MAX, offset);
        }
        return mask;
    }
}