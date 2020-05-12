#include "mct.h"

#include <Windows.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>

using namespace cv;
using namespace std;

namespace mct
{
    void showImage(const Mat& image, string name, float size)
    {
        Mat img_out = image.clone();
        resize(image, img_out, Size(), size, size, INTER_LINEAR_EXACT);
        imshow(name, img_out);
        waitKey(0);
    }

    void showImageWithMouse(const Mat& image, float size)
    {
        Mat img_out = image.clone();
        resize(image, img_out, Size(), size, size, INTER_LINEAR_EXACT);
        namedWindow("mouse");
        setMouseCallback("mouse", [](int event, int x, int y, int flags, void* img_in) {
                char text[100];
                Mat img = *(Mat*)img_in;
                Mat img2 = img.clone();

                float p = img.at<float>(y, x);
                snprintf(text, 100, "x=%d, y=%d Val=%f", x, y, p);

                rectangle(img2, Rect(0, 0, 200, 20), Scalar(0), FILLED);
                putText(img2, text, Point(5, 15), FONT_HERSHEY_PLAIN, 1.0, Scalar(255));
                imshow("mouse", img2); 
            }, (void*)&img_out);
        imshow("mouse", img_out);
        waitKey(0);
    }

    int getopt(int argc, char* const argv[], const char* optarg)
    {
        return 0;
    }

    Mat toGrayScale(const Mat& img_in)
    {
        Mat img_gray = img_in.clone();
        if (img_in.channels() == 3) cvtColor(img_in, img_gray, COLOR_RGB2GRAY, 1);
        return img_gray;
    }

    chrono::steady_clock::time_point startTimer(void)
    {
        return chrono::high_resolution_clock::now();
    }

    double stopTimer(std::chrono::steady_clock::time_point startTime)
    {
        return chrono::duration<double>(chrono::high_resolution_clock::now() - startTime).count();
    }

    void drawDottedRect(Mat& img, Rect r, Scalar dot_color, int dot_length, int dot_height, int dot_spacing)
    {
        int step = dot_length + dot_spacing;

        for (int x = r.x; x <= r.br().x; x += step)
            rectangle(img, Rect(x, 0, dot_length, dot_height), dot_color, FILLED);
        for (int x = r.br().x; x >= r.x; x -= step)
            rectangle(img, Rect(Point(x, r.br().y), Point(x - dot_length, r.br().y - dot_height)), dot_color, FILLED);
        for (int y = r.y; y <= r.br().y; y += step)
            rectangle(img, Rect(r.br().x - dot_height, y, dot_height, dot_length), dot_color, FILLED);
        for (int y = r.br().y; y >= 0; y -= step)
            rectangle(img, Rect(Point(0, y), Point(0 + dot_height, y - dot_length)), dot_color, FILLED);
    }

    double pointDist(Point2f a, Point2f b)
    {
        return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    }

    double pointDist(Point a, Point b)
    {
        return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    }

    /*
        Checks two Rect
        return 0 if no intersect, 1 if r1 fully contain r2, -1 if r2 fully contain r1, 2 for partial
    */
    int compareRect(Rect r1, Rect r2)
    {
        if (r1.x > r2.br().x || r1.br().x < r2.x
            || r1.y > r2.br().y || r1.br().y < r2.y)
        {
            return 0;
        }

        if (r1.x <= r2.x && r1.br().x >= r2.br().x
            && r1.y <= r2.y && r1.br().y >= r2.br().y)
        {
            return 1;
        }
        else if (r1.x >= r2.x && r1.br().x <= r2.br().x
            && r1.y >= r2.y && r1.br().y <= r2.br().y)
        {
            return -1;
        }

        return 2;
    }

    /*
        Checks two contour
        return 0 if no intersect, 1 if r1 fully contain r2, -1 if r2 fully contain r1, 2 for partial
        TODO: Make it more elegant
    */
    int compareContour(vector<Point>& c1, vector<Point>& c2)
    {
        Rect r1 = boundingRect(c1);
        Rect r2 = boundingRect(c2);
        Mat canvas1(Size(max(r1.width, r2.width), max(r1.height, r2.height)), CV_8UC1, Scalar(0));
        Mat canvas2(Size(max(r1.width, r2.width), max(r1.height, r2.height)), CV_8UC1, Scalar(0));
        drawContours(canvas1, { c1 }, -1, Scalar(255), FILLED);
        drawContours(canvas2, { c2 }, -1, Scalar(255), FILLED);
        bitwise_and(canvas1, canvas2, canvas1);
        
        vector<Point> result;
        findContours(canvas1, result, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (result.size() == 0)
        {
            return 0;
        }
        else if (contourArea(result) == contourArea(c2))
        {
            return 1;
        }
        else if (contourArea(result) == contourArea(c1))
        {
            return -1;
        }
        else
        {
            return 2;
        }
    }

    /*
       Computes min distance from point to edge of rect
    */
    int minEdgeDist(Rect r, Point p)
    {
        return min(r.width / 2 - abs((r.x + r.width / 2) - p.x), r.height / 2 - abs((r.y + r.height / 2) - p.y));
    }

    // Checks background (border) color of image
    Scalar bgColor(const Mat& img, int borderWidth)
    {
        Rect borderMask(borderWidth, borderWidth, img.cols - borderWidth - 1, img.rows - borderWidth - 1);
        Mat img_mask = img.clone();
        rectangle(img_mask, borderMask, Scalar(0), FILLED);
        return mean(img, img_mask);
    }

    int tagImage(const Mat& display_image)
    {
        while (true)
        {
            imshow("Sorter", display_image);
            waitKey(0);
            if (GetKeyState(VK_LEFT) & 0x8000)
            {
                return 0;
            }
            else if (GetKeyState(VK_RIGHT) & 0x8000)
            {
                return 1;
            }
            else if (GetKeyState(VK_UP) & 0x8000)
            {
                return -1;
            }
        }
        return 0;
    }

    vector<bool> manualSorter(const Mat& image, vector<Rect>& rect, const Size& display_size)
    {
        namedWindow("Sorter");
        vector<bool> result(rect.size());
        for (int i = 0; i <= rect.size(); i++)
        {
            if (i == rect.size())
            {
                cout << "Press Up to undo; Press other key to confirm" << endl;
                waitKey(0);
                if (GetKeyState(VK_UP) & 0x8000)
                {
                    i--;
                }
                else
                {
                    break;
                }
            }
        START:
            cout << "Img " << i << " of " << rect.size() << string(20, ' ') << "\r";

            Size sz = display_size;
            if (display_size.width < 0) sz = rect[i].size();
            Mat bbl(image, rect[i]);
            Mat dst;
            resize(bbl, dst, sz, 0, 0, 0);
            switch (tagImage(dst))
            {
            case 0:
                result[i] = false;
                break;
            case 1:
                result[i] = true;
                break;
            case -1:
                if (i > 0) i--;
                goto START;
            }
        }
        destroyWindow("Sorter");
        return result;
    }


    void manualFrameSorter(const Mat& image, vector<Frame>& frames)
    {
        vector<Rect> rect(frames.size());
        for (int i = 0; i < frames.size(); i++)
        {
            rect[i] = frames[i].box;
        }

        vector<bool> tag = manualSorter(image, rect, Size(-1, -1));
        for (int i = 0; i < frames.size(); i++)
        {
            frames[i].is_frame = tag[i];
        }
    }
};