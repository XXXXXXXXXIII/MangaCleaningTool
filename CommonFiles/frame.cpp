#include "frame.h"
#include "mct.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

using namespace std;
using namespace cv;

namespace mct
{
    struct cmpRect2 {
        bool operator() (const Rect& lhs, const Rect& rhs) const
        {
            return lhs.area() > rhs.area();
        }
    };

    bool cmpRect(const pair<Rect, Point>& lhs, const pair<Rect, Point>& rhs)
    {
        return lhs.first.area() > rhs.first.area();
    }

    /*
        Extract frame, returns list of pair of individual frame mask and its rect relative to img
        Derived from: https://ieeexplore.ieee.org/document/5501698
        @param img_bin: greyscale of the image, CV_8UC1
        @param frame_mask: returned mask for panels, CV_8UC1
    */
    vector<Frame> extractFrame(const Mat& img)
    {
        const uchar BACK_COLOR_L = 0, BACK_COLOR_H = 1;
        const uchar FRAME_COLOR_L = 128, FRAME_COLOR_H = 129;
        const uchar CONTOUR_COLOR_L = 254, CONTOUR_COLOR_H = 255;

        Mat img_bin;
        Mat img_tmp;
        bilateralFilter(img, img_bin, 5, 50, 50); // Remove noise on image //TODO: Find better simga value
        threshold(img_bin, img_bin, -1, 255, THRESH_OTSU); // Good for noisy image, however removes details on gradients
        bitwise_not(img_bin, img_bin);
        //threshold(img, img_bin, 240, 255, THRESH_BINARY); // Good for clean image; will cause frames to merge on dirty images
        //frame_mask = Mat(img_bin.size(), CV_8UC1, Scalar(255));

        // Find external contour
        vector<vector<Point>> ext_contour, hull_contour;
        drawDottedRect(img_bin, Rect(0, 0, img_bin.cols, img_bin.rows), Scalar(255), 2, 4, 4);
        findContours(img_bin, ext_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        drawContours(img_bin, ext_contour, -1, Scalar(CONTOUR_COLOR_L), FILLED, 8);
        vector<pair<Rect, Point>> ext_box, frame_box;
        for (auto c : ext_contour)
        {
            Rect box = boundingRect(c);
            ext_box.push_back(pair<Rect, Point>(box, c[0])); //TODO: Find better seed points
        }
        sort(ext_box.begin(), ext_box.end(), cmpRect);

        // connect small components
        for (int e = 0; e < ext_box.size(); e++)
        {
            bool merged = false;
            vector<int> partial;
            for (int f = 0; f < frame_box.size(); f++)
            {
                int state = compareRect(frame_box[f].first, ext_box[e].first);
                if (state == 1)
                {
                    line(img_bin, ext_box[e].second, frame_box[f].second, Scalar(CONTOUR_COLOR_L), 1);
                    merged = true;
                }
                else if (state == 2 && frame_box[f].first.area() < 50) //TODO: Use dynamic area?
                {
                    partial.push_back(f);
                }
            }

            if (!merged)
            {
                if (partial.empty())
                {
                    frame_box.push_back(ext_box[e]);
                    continue;
                }

                for (auto p : partial)
                {
                    line(img_bin, ext_box[e].second, frame_box[p].second, Scalar(CONTOUR_COLOR_L), 1);

                }
            }
        }
        findContours(img_bin, ext_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // find hull
        for (auto c : ext_contour)
        {
            vector<int> hull_idx;
            vector<Point> hull;
            vector<Vec4i> defects;
            convexHull(c, hull_idx);
            if (!isContourConvex(c))
            {
                convexHull(c, hull);
                hull_contour.push_back(hull);
                continue;
            }

            convexityDefects(c, hull_idx, defects);
            if (defects.empty())
            {
                convexHull(c, hull);
                hull_contour.push_back(hull);
                continue;
            }

            int idx = defects[0][0];
            for (auto d : defects)
            {
                while (idx <= d[0])
                {
                    hull.push_back(c[idx]);
                    idx++;
                    if (idx >= c.size())
                    {
                        idx = 0;
                    }
                }

                double avgdist = 0; // Remove hulls that are too far from original contour
                {
                    const int sample_count = 100;
                    Point2f p = c[d[0]];
                    Point2f dir = (c[d[1]] - c[d[0]]) / sample_count;
                    for (int i = 0; i < sample_count; i++)
                    {
                        avgdist += pointPolygonTest(c, p, true);
                        p += dir;
                    }
                    avgdist /= sample_count;
                }

                if (avgdist > -10.0 || pointDist(c[d[0]], c[d[1]]) < 5) // TODO: use relative distance??
                {
                    idx = d[1];
                }
                //cout << c.size() << d << pointDist(c[d[0]], c[d[1]]) << "  " << contourArea(vector<Point>{c[d[0]], c[d[1]], c[d[2]]}) << "   " << avgdist << endl;
                //line(frame_mask, c[d[0]], c[d[1]], Scalar(0), 3);
                //showImage(frame_mask);
            }

            while (idx != defects[0][0])
            {
                hull.push_back(c[idx]);
                idx++;
                if (idx >= c.size())
                {
                    idx = 0;
                }
            }
            hull_contour.push_back(hull);
        }
        drawContours(img_bin, hull_contour, -1, Scalar(CONTOUR_COLOR_L), FILLED, 8);

        vector<Frame> frames;
        for (auto c : hull_contour)
        {
            if (contourArea(c) < 10) continue;
            Frame f = {img.size(), boundingRect(c), c};
            frames.push_back(f);
        }

        return frames;
    }

    void cleanFrame(Mat& img, const vector<Frame>& frames, uchar back_color)
    {
        Mat frame_mask = createFrameMask(img.size(), frames);
        Mat frame_mask_inv;
        bitwise_not(frame_mask, frame_mask_inv);
        bitwise_and(img, frame_mask, img);
        bitwise_or(img, Scalar(back_color), img, frame_mask_inv);
    }

    Mat createFrameMask(const Size& size, const vector<Frame>& frames)
    {
        Mat mask(size, CV_8UC1, Scalar(0));
        for (Frame f : frames)
        {
            if (!f.is_frame) continue;
            drawContours(mask, vector<vector<Point>>{ f.contour }, -1, Scalar(255), FILLED);
        }
        return mask;
    }
};

/*
    Common Cases:
        - regular frame with black borders
            .: trace exterior contour
        - regular frame with holes on border
            .: convex hull, remove large convex defects
        - frame that borders the edge of the page
            .: seal entire image with dotted lines
        - connected frames (irregular shape)
            .: Do not cut.
    Edge cases:
        - black/pattern background
            .: TODO
*/