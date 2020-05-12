#include "text.h"
#include "mct.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

namespace mct
{
    /*
        Reference: https://github.com/aperrault/DetectText/blob/master/TextDetection.cpp
        @param image: CV_8UC1
        @param angle: PI/angle
        @param dark_text: ray cast over dark
    */
    Mat strokeWidthTransform(const Mat& image, float angle, bool dark_text)
    {
        struct Ray
        {
            Point p;
            Point q;
            vector<Point> r;
        };

        Mat img_bin(image.size(), CV_8UC1);
        Mat img_float(image.size(), CV_32FC1);
        Mat img_edge(image.size(), CV_32FC1);
        Mat img_gradX(image.size(), CV_32FC1);
        Mat img_gradY(image.size(), CV_32FC1);
        Mat img_swt(image.size(), CV_32FC1, Scalar(-1));

        threshold(image, img_bin, 240, 255, THRESH_BINARY);
        img_bin.convertTo(img_float, CV_32FC1, 1. / 255.);

        GaussianBlur(img_bin, img_bin, Size(5, 5), 0);
        Canny(img_bin, img_edge, 128, 255); // Blur seem to run faster

        Scharr(img_float, img_gradX, -1, 1, 0);
        Scharr(img_float, img_gradY, -1, 0, 1);
        GaussianBlur(img_gradX, img_gradX, Size(3, 3), 0);
        GaussianBlur(img_gradY, img_gradY, Size(3, 3), 0);

        vector<Ray> rays;
        int counter = 0;

        // Compute stroke width
        for (int y = 0; y < img_edge.rows; y++)
        {
            const uchar* rptr = img_edge.ptr<const uchar>(y);
            const float* Gx_ptr = img_gradX.ptr<const float>(y);
            const float* Gy_ptr = img_gradY.ptr<const float>(y);
            for (int x = 0; x < img_edge.cols; x++)
            {
                if (*rptr > 0)
                {
                    float Gx = Gx_ptr[x];
                    float Gy = Gy_ptr[x];
                    float Gmag = sqrt((Gx * Gx) + (Gy * Gy));
                    if (dark_text)
                    {
                        Gx = -Gx / Gmag;
                        Gy = -Gy / Gmag;
                    }
                    else
                    {
                        Gx = Gx / Gmag;
                        Gy = Gy / Gmag;
                    }

                    int dx = Gx / abs(Gx);
                    int dy = Gy / abs(Gy);
                    int qx = x;
                    int qy = y;

                    Ray currRay;
                    vector<Point> path;
                    path.push_back(Point(x, y));
                    currRay.p = Point(x, y);

                    while (true)
                    {
                        if (abs(Gx) > abs(Gy))
                        {
                            qx += dx;
                            if (y + dy * floor(abs((qx - x) / Gx * Gy)) != qy)
                            {
                                qy += dy;
                                qx = x + dx * floor(abs((qy - y) / Gy * Gx));
                            }
                        }
                        else
                        {
                            qy += dy;
                            if (x + dx * floor(abs((qy - y) / Gy * Gx)) != qx)
                            {
                                qx += dx;
                                qy = y + dy * floor(abs((qx - x) / Gx * Gy));
                            }
                        }
                        path.push_back(Point(qx, qy));

                        if (qx < 0 || qy < 0 || (qx >= img_edge.cols) || (qy >= img_edge.rows))
                        {
                            break;
                        }

                        if (img_edge.at<uchar>(qy, qx) > 0)
                        {
                            float qGx = img_gradX.at<float>(qy, qx);
                            float qGy = img_gradY.at<float>(qy, qx);
                            float qGmag = sqrt((qGx * qGx) + (qGy * qGy));
                            if (dark_text)
                            {
                                qGx = -qGx / qGmag;
                                qGy = -qGy / qGmag;
                            }
                            else
                            {
                                qGx = qGx / qGmag;
                                qGy = qGy / qGmag;
                            }
                            if (acos(Gx * -qGx + Gy * -qGy) < CV_PI / angle)
                            {
                                currRay.q = Point(qx, qy);
                                currRay.r = path;
                                float dist = (float)sqrt((qx - x) * (qx - x) + (qy - y) * (qy - y));
                                for (Point& p : path)
                                {
                                    if (img_swt.at<float>(p) > dist || img_swt.at<float>(p) < 0)
                                    {
                                        img_swt.at<float>(p) = dist;
                                    }
                                }
                                counter++;
                                rays.push_back(currRay);
                                //line(img_swt, Point(x, y), Point(qx, qy), Scalar(0.5));
                            }
                            break;
                        }
                        //img_swt.at<uchar>(qy, qx) = 150;
                    }
                }
                rptr++;
            }
        }

        //cout << "Number of rays: " << counter << endl;

        // Median
        for (Ray& r : rays)
        {
            vector<float> widths;
            for (Point& p : r.r)
            {
                widths.push_back(img_swt.at<float>(p));
            }

            sort(widths.begin(), widths.end());
            float median = widths[widths.size() / 2];

            for (Point& p : r.r)
            {
                if (img_swt.at<float>(p) > median)
                {
                    img_swt.at<float>(p) = median;
                }
            }
        }

        return img_swt;
    }

    vector<Point> connectedNeighbor(const Mat& img, const Point& p, float loFillRatio, float hiFillRatio, int scan_radius)
    {
        vector<Point> result;
        float value = img.at<float>(p);
        for (int r = p.y - scan_radius; r <= p.y + scan_radius; r++)
        {
            if (r < 0 || r >= img.rows) continue;
            for (int c = p.x - scan_radius; c <= p.x + scan_radius; c++)
            {
                if (c < 0 || c >= img.cols) continue;

                if (img.at<float>(r, c) <= value * hiFillRatio &&
                    img.at<float>(r, c) >= value / loFillRatio)
                {
                    result.push_back(Point(c, r));
                }
            }
        }
        return result;
    }

    /*
    Modified two-pass CCL algorithm for swt

    //TODO: Optimize

    @param img_swt: CV32FC1
    @param loFillDiff: lower bound for connected pixel, as divisor
    @param hiFillDiff: upper bound for connected pixel, as multiplier
    @param scan_radius: look for connected pixels within this radius
*/
    Mat connectSWTComponents(const Mat& img_swt, float loFillDiff, float hiFillDiff, int scan_radius)
    {
        Mat img_ccl(img_swt.size(), CV_32FC1, Scalar(-1));
        int counter = 0;
        vector<int> labels;
        //vector<set<int>> labelGraph;
        vector<int> labelGraph;

        for (int r = 0; r < img_swt.rows; r++)
        {
            const float* rtpr = img_swt.ptr<float>(r);
            for (int c = 0; c < img_swt.cols; c++)
            {
                if (rtpr[c] > 0.0)
                    //&& img_ccl.at<float>(r, c) < 0)
                {
                    vector<Point> neighbor = connectedNeighbor(img_swt, Point(c, r), loFillDiff, hiFillDiff, scan_radius);

                    int minLabel = INT_MAX;
                    vector<int> neighborLabel;
                    for (Point n : neighbor)
                    {
                        int v = img_ccl.at<float>(n);
                        if (v >= 0)
                        {
                            if (v < minLabel) minLabel = v;
                            neighborLabel.push_back(v);
                        }
                    }

                    if (minLabel == INT_MAX)
                    {
                        for (Point n : neighbor)
                        {
                            if (img_ccl.at<float>(n) < 0)
                            {
                                img_ccl.at<float>(n) = counter;
                            }
                        }
                        labelGraph.push_back(counter);
                        counter++;
                    }
                    else
                    {
                        for (int i : neighborLabel)
                        {
                            if (minLabel <= i)
                            {
                                labelGraph[i] = minLabel;
                            }
                            else
                            {
                                labelGraph[minLabel] = i;
                            }
                        }
                        for (Point n : neighbor)
                        {
                            if (img_ccl.at<float>(n) < 0)
                            {
                                img_ccl.at<float>(n) = minLabel;
                            }
                        }
                    }

                    //cout << Point(c, r) << img_ccl.at<float>(r, c) << "   " << neighbor.size() << endl;
                }
            }
        }

        for (int i = 0; i < labelGraph.size(); i++)
        {
            int j = i;
            while (labelGraph[j] < j)
            {
                j = labelGraph[j];
            }
            labelGraph[i] = j;
            //while (labelGraph[j].size() > 0 && *labelGraph[j].begin() < j)
            //{
            //    j = *labelGraph[j].begin();
            //}
            ////labelGraph[i].clear();
            //labelGraph[i].insert(j);
        }

        for (int r = 0; r < img_swt.rows; r++)
        {
            float* rptr = (float*)img_ccl.ptr<float>(r);
            for (int c = 0; c < img_swt.cols; c++)
            {
                float label = rptr[c];
                if (label >= 0)
                {
                    rptr[c] = labelGraph[label];
                }
            }
        }

        //showImage(img_swt, "image", 1, true);
        return img_ccl;
    }

    /*
        Filter letter candidates: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/1509.pdf
    */
    vector<Rect> textCandidateFilter(const Mat& img_ccl)
    {
        const int MIN_TEXT_LENGTH = 7;
        const int MAX_TEXT_LENGTH = 200;
        const int MAX_TEXT_ASPECT = 5;
        const float MIN_TEXT_FILL = 0.4;
        const int MAX_TEXT_OVERLAP = 20;

        vector<Rect> elected;
        vector<Rect> candidates;
        vector<int> fillArea;
        vector<float> avgWidth;

        for (int r = 0; r < img_ccl.rows; r++)
        {
            const float* rtpr = img_ccl.ptr<const float>(r);
            for (int c = 0; c < img_ccl.cols; c++)
            {
                long label = (int)rtpr[c];
                if (label < 0) continue;

                if (candidates.size() <= label)
                {
                    candidates.resize(label + 1);
                    fillArea.resize(label + 1, -1);
                    candidates[label] = Rect(c, r, 1, 1);
                    fillArea[label] = 1;
                }
                else
                {
                    fillArea[label]++;
                    if (candidates[label].x > c)
                        candidates[label].x = c;
                    else if (candidates[label].br().x < c)
                        candidates[label].width = c - candidates[label].x;

                    if (candidates[label].y > r)
                        candidates[label].y = r;
                    else if (candidates[label].br().y < r)
                        candidates[label].height = r - candidates[label].y;
                }

            }
        }

        for (int i = 0; i < candidates.size(); i++)
        {
            Rect r = candidates[i];
            int filled = fillArea[i];

            // Size, aspect ratio, 
            if (filled < 0
                || max(r.width, r.height) > MAX_TEXT_LENGTH
                || min(r.width, r.height) < MIN_TEXT_LENGTH
                || max(r.width / r.height, r.height / r.width) > MAX_TEXT_ASPECT
                || (r.area() * MIN_TEXT_FILL) > filled)
            {
                continue;
            }

            int overlap = 0;
            for (int y = r.y; y <= r.br().y; y++)
            {
                const float* swt_ptr = img_ccl.ptr<const float>(y);
                for (int x = r.x; x <= r.br().x; x++)
                {
                    if ((int)swt_ptr[x] == -1) continue;
                    if (swt_ptr[x] != i) overlap++;
                }
            }

            // Overlaps
            if (overlap > MAX_TEXT_OVERLAP)
            {
                continue;
            }

            elected.push_back(r);
        }
        return elected;
    }

	vector<Rect> findTextCandidate(const Mat& image)
	{
        Mat img_swt = strokeWidthTransform(image, 2.0, true);
        // 5, 5 seem to work the best
        Mat img_ccl = connectSWTComponents(img_swt, 5, 5, 1); //TODO: Combine broken strokes into one character (i.e. こ)

        //showImage(img_swt);
        //showImage(img_ccl);

        //TODO: train text detector?
        vector<Rect> boxes = textCandidateFilter(img_ccl);

        //for (auto& b : boxes)
        //{
        //    rectangle(image, b, Scalar(120), 2);
        //}
        //showImage(image);

        return boxes;
	}
}