#include "frame.h"
#include "mct.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

namespace mct
{
    /*
        Extract frame, returns list of pair of individual frame mask and its rect relative to img
        Derived from: https://ieeexplore.ieee.org/document/5501698
        @param img: greyscale image, CV_8UC1
        TODO: Improve frame extraction
    */
    vector<Frame> extractFrame(const Mat& img)
    {
        const uchar BACK_COLOR_L = 0, BACK_COLOR_H = 1;
        const uchar FRAME_COLOR_L = 128, FRAME_COLOR_H = 129;
        const uchar CONTOUR_COLOR_L = 254, CONTOUR_COLOR_H = 255;

        if (false)
        {
            Ptr<BoostFrameClassifier> frame_boost = new BoostFrameClassifier();
            Mat img_bin;
            threshold(img, img_bin, -1, 255, THRESH_OTSU);
            if (bgColor(img)[0] > 128)
            {
                bitwise_not(img_bin, img_bin);
            }
            vector<vector<Point>> ext_contour, hull_contour;
            findContours(img_bin, ext_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            drawContours(img_bin, ext_contour, -1, Scalar(CONTOUR_COLOR_H), FILLED);
            vector<Frame> frames(ext_contour.size());
            for (int i = 0; i < frames.size(); ++i)
            {
                frames[i].contour = ext_contour[i];
                frames[i].box = boundingRect(ext_contour[i]);
                frames[i].is_frame = true;
                frames[i].page = img.size();
            }

            sort(frames.begin(), frames.end(), [](const Frame& lhs, const Frame& rhs)
                {
                    return lhs.box.area() > rhs.box.area();
                });

            struct Line
            {
                Point start;
                Point end;
            };

            vector<Line> lines;
            for (auto& f : frames)
            {
                vector<Point> hull;
                convexHull(f.contour, hull);

                int maxX = 0, maxY = 0, minX = img.cols, minY = img.rows;

                for (auto& h : hull)
                {
                    if (h.x > maxX) maxX = h.x;
                    if (h.x < minX) minX = h.x;
                    if (h.y > maxY) maxY = h.y;
                    if (h.y < minY) minY = h.y;
                    //lines.push_back(Line{ Point(h.x, 0), Point(h.x, img_bin.rows) });
                    //lines.push_back(Line{ Point(0, h.y), Point(img_bin.cols, h.y) });
                }

                lines.push_back(Line{ Point(minX - 1, 0), Point(minX - 1, img_bin.rows) });
                lines.push_back(Line{ Point(maxX + 1, 0), Point(maxX + 1, img_bin.rows) });
                lines.push_back(Line{ Point(0, minY - 1), Point(img_bin.cols, minY - 1) });
                lines.push_back(Line{ Point(0, maxY + 1), Point(img_bin.cols, maxY + 1) });
            }

            vector<Line> segments;
            for (auto& l : lines)
            {
                vector<Point> seg{ l.start, l.end };
                for (auto& l2 : lines)
                {
                    Rect r(0, 0, img.cols, img.rows);
                    Point x = lineIntersect(l.start, l.end, l2.start, l2.end);
                    if (!r.contains(x) || l.start == l2.start) continue;

                    seg.push_back(x);
                    
                    //circle(img_bin, x, 5, Scalar(180), 2);
                }

                sort(seg.begin(), seg.end(), [](const Point& lhs, const Point& rhs)
                    {
                        if (lhs.x == rhs.x) return lhs.y < rhs.y;
                        else return lhs.x < rhs.x;
                    });

                for (int i = 1; i < seg.size(); i++)
                {
                    if (seg[i - 1] == seg[i]) continue;
                    segments.push_back(Line{ seg[i - 1], seg[i] });
                }
                //line(img_bin, l.start, l.end, Scalar(180), 1);
                //circle(img_bin, l.start, 2, Scalar(180));
                //circle(img_bin, l.end, 2, Scalar(180));
            }

            for (auto& l : segments)
            {
                Point2f p = l.start;
                Point2f dir = Point2f(l.end - l.start) / pointDist(l.end, l.start);
                //cout << dir << endl;
                for (int i = 0; i < ceil(pointDist(l.end, l.start)); i++)
                {
                    if (p.x < 0 || p.x >= img_bin.cols
                        || p.y < 0 || p.y >= img_bin.rows) break;
                    if (img_bin.at<uchar>(p) > 0)
                    {
                        l.start = Point(-1, -1);
                        l.end = Point(-1, -1);
                        break;
                    }
                    p += dir;
                }
            }

            for (auto& l : segments)
            {
                line(img_bin, l.start, l.end, Scalar(120), 2);
            }

            //showImage(img);
            //showImage(img_bin);
            
            //TODO: contour line tracer

            //frame_boost->classifyFrame(frames);

            //vector<Frame> result;
            //frames.erase(remove_if(frames.begin(), frames.end(), 
            //    [](const Frame& f) { return !f.is_frame; }), frames.end());

            //img_bin = Mat::zeros(img.size(), CV_8UC1);
            //for (auto& f : frames)
            //{
            //    bool intersect = false;
            //    for (auto& f2 : frames)
            //    {
            //        if (f2.box != f.box && !(f2.box & f.box).empty())
            //        {
            //            if ((f2.box & f.box) == f.box) // Contained
            //            {
            //                intersect = true;
            //                break;
            //            }
            //            else if ((f2.box & f.box) == f2.box) // Contains
            //            {
            //                continue;
            //            }
            //            else // Partial
            //            {
            //                intersect = true;
            //                //TODO: COnnect the dots
            //            }
            //        }
            //    }
            //    if (!intersect)
            //    {
            //        drawContours(img_bin, vector<vector<Point>>{ f.contour }, -1, Scalar(CONTOUR_COLOR_L), FILLED);
            //        result.push_back(f);
            //    }
            //}
                //showImage(img_bin);
        }

        Mat img_bin;
        Mat img_tmp;
        threshold(img, img_bin, -1, 255, THRESH_OTSU); // Good for noisy image, however removes details on gradients
        bitwise_not(img_bin, img_bin);
        //threshold(img, img_bin, 240, 255, THRESH_BINARY); // Good for clean image; will cause frames to merge on dirty images
        //frame_mask = Mat(img_bin.size(), CV_8UC1, Scalar(255));

        // Find external contour
        vector<vector<Point>> ext_contour, hull_contour;
        drawDottedRect(img_bin, Rect(0, 0, img_bin.cols, img_bin.rows), Scalar(255), 2, 4, 4);
        findContours(img_bin, ext_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        drawContours(img_bin, ext_contour, -1, Scalar(CONTOUR_COLOR_L), FILLED, 8);
        vector<pair<Rect, Point>> ext_box, frame_box;
        for (auto& c : ext_contour)
        {
            Rect box = boundingRect(c);
            ext_box.push_back(pair<Rect, Point>(box, c[0])); //TODO: Find better seed points
        }
        sort(ext_box.begin(), ext_box.end(), [](const pair<Rect, Point>& lhs, const pair<Rect, Point>& rhs)
            {
                return lhs.first.area() > rhs.first.area();
            });

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
        for (auto& c : ext_contour)
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
            for (auto& d : defects)
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
        for (auto& c : hull_contour)
        {
            if (contourArea(c) < 10) continue;
            Frame f = {img.size(), boundingRect(c), c};
            frames.push_back(f);
        }

        return frames;
    }

    void cleanFrame(Mat& img, const vector<Frame>& frames, uchar back_color) //TOOD: Update function
    {
        Mat frame_mask = createFrameMask(img.size(), frames);
        Mat frame_mask_inv;
        bitwise_not(frame_mask, frame_mask_inv);
        bitwise_and(img, frame_mask, img);
        bitwise_or(img, Scalar(back_color), img, frame_mask_inv);
    }

    Mat createFrameMask(const Size& size, const vector<Frame>& frames, const uchar back_color, const uchar mask_color)
    {
        Mat mask(size, CV_8UC1, Scalar(back_color));
        for (const Frame& f : frames)
        {
            if (!f.is_frame) continue;
            drawContours(mask, vector<vector<Point>>{ f.contour }, -1, Scalar(mask_color), FILLED);
        }
        return mask;
    }

    // Erase patterned frame from image for frame extraction
    // TODO: Finish this, and implemment pattern inpainting
    Mat erasePatternFrame(const Mat& image)
    {
        Mat img_clone = image.clone();
        for (int i = 4; i < 5; i++)
        {
            Mat patch(image, Rect(1, 1, i, i));
            Mat result;
            matchTemplate(image, patch, result, TM_SQDIFF_NORMED);
            normalize(result, result, 0, 1, NORM_MINMAX);
            showImage(result);
            matchTemplate(image, patch, result, TM_SQDIFF);
            normalize(result, result, 0, 1, NORM_MINMAX);
            showImage(result);
            matchTemplate(image, patch, result, TM_CCORR);
            normalize(result, result, 0, 1, NORM_MINMAX);
            showImage(result);
            matchTemplate(image, patch, result, TM_CCORR_NORMED);
            normalize(result, result, 0, 1, NORM_MINMAX);
            showImage(result);
            //showImage(result);
            resize(result, result, img_clone.size());
            //showImage(result);
            result.convertTo(result, CV_8UC1, 255);
            //showImage(result);
            Mat mask;
            threshold(result, result, -1, 255, THRESH_OTSU);
            bitwise_not(result, result);
            showImage(result);
            bitwise_or(img_clone, Scalar(255), img_clone, result);
            cout << i << endl;
            //showImage(patch, "image", 1);
            //showImage(result, "image", 0.4);
            //showImage(img_clone);
        }
        return img_clone;
    }

    BoostFrameClassifier::BoostFrameClassifier(std::string str_model)
    {
        this->frameClassifier = Boost::loadFromString<Boost>(cv::String(str_model));
    }

    void BoostFrameClassifier::classifyFrame(Frame& frame)
    {
        Mat result;
        this->frameClassifier->predict(frame.toInputData(), result);
        frame.is_frame = result.at<float>(0, 0) > 0;
    }

    void BoostFrameClassifier::classifyFrame(vector<Frame>& frames)
    {
        for (int i = 0; i < frames.size(); i++)
        {
            Mat result;
            this->frameClassifier->predict(frames[i].toInputData(), result);
            frames[i].is_frame = result.at<float>(0, 0) > 0;
        }
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


/* // Discard
    vector<Frame> extractFrame(const Mat& img)
    {
        const uchar BACK_COLOR_L = 0, BACK_COLOR_H = 1;
        const uchar FRAME_COLOR_L = 128, FRAME_COLOR_H = 129;
        const uchar CONTOUR_COLOR_L = 254, CONTOUR_COLOR_H = 255;


        {
            Mat temp;
            blur(img, temp, Size(5, 5));
            Canny(temp, temp, 200, 128);
            vector<vector<Point>> ext_contour, hull_contour;
            findContours(temp, ext_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            drawContours(temp, ext_contour, -1, Scalar(180), 3);
            showImage(temp);


            //bilateralFilter(img, temp, 5, 50, 50);
            ////showImage(temp);
            //inRange(temp, Scalar(0), Scalar(25), temp);

            //Ptr<MSER> mser = MSER::create();
            //vector<vector<Point>> pts;
            //vector<Rect> box;
            //mser->detectRegions(img, pts, box);
            //for (auto& b : box)
            //{
            //    rectangle(temp, b, Scalar(180), 4);
            //}
            //showImage(temp);
        }

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
        for (auto& c : ext_contour)
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
        for (auto& c : ext_contour)
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
            for (auto& d : defects)
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
        for (auto& c : hull_contour)
        {
            if (contourArea(c) < 10) continue;
            Frame f = {img.size(), boundingRect(c), c};
            frames.push_back(f);
        }

        return frames;
    }
*/