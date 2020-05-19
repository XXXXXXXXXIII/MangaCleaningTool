#include "text.h"
#include "mct.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

namespace mct
{
    vector<Text> MSERTextDetector::detectTextLine(const Mat& image)
    {
        Mat img_bin;
        bitwise_not(image, img_bin);
        threshold(img_bin, img_bin, 64, 255, THRESH_OTSU);
        bitwise_not(img_bin, img_bin);

        vector<Text> text;
        vector<vector<Point>> point;
        vector<Rect> box;
        mser->detectRegions(img_bin, point, box);

        Mat img_swt = strokeWidthTransform(image);

        //vector<float> score(box.size());
        for (int i = 0; i < box.size(); i++)
        {
            if (box[i].area() > image.size().area() / 2) continue;
            Text t = {box[i], 0, 0};
            for (int r = box[i].y; r < box[i].y + box[i].height; r++)
            {
                for (int c = box[i].x; c < box[i].x + box[i].width; c++)
                {
                    if (img_swt.at<float>(r, c) > 0)
                    {
                        t.fill_area++;
                        t.avg_width += img_swt.at<float>(r, c);
                    }
                }
            }
            t.avg_width /= t.fill_area;
            text.push_back(t);
            //score[i] = area[i] / box[i].area();
        }

        vector<Text> text_line_vert, text_line_hori, text_line_final;

        //TODO: Find a better solution
        if (image.size().aspectRatio() > 1)
        {
            text_line_hori = joinTextLine(text, ORIENTATION::HORIZONTAL);
        }
        else
        {
            text_line_vert = joinTextLine(text, ORIENTATION::VERTICAL);
        }

        // Merge all lines
        vector<bool> do_keep(text_line_hori.size(), true);
        for (auto& lv : text_line_vert)
        {
            bool keep = true;
            int i = 0;
            for (auto& lh : text_line_hori)
            {
                if (!(lv.box & lh.box).empty())
                {
                    if (lh.box.width / lh.box.height > lv.box.height / lv.box.width
                        && lh.box.width > lv.box.height)
                    {
                        keep = false;
                        break;
                    }
                    else
                    {
                        do_keep[i] = false;
                    }
                }
                ++i;
            }
            if (keep)
            {
                text_line_final.push_back(lv);
            }
        }

        for (int i = 0; i < do_keep.size(); i++)
        {
            if (do_keep[i])
            {
                text_line_final.push_back(text_line_hori[i]);
            }
        }

        //for (auto& l : text_line_final)
        //{
        //    rectangle(image, l.box, Scalar(180), 4);
        //    cout << l.avg_width << " , " << l.fill_area << endl;
        //showImage(image);
        //}

        return text_line_final;
    }

    vector<Text> MSERTextDetector::joinTextLine(vector<Text>& text, ORIENTATION o)
    {
        Point shift;
        Size scale;
        if (o == ORIENTATION::VERTICAL)
        {
            sort(text.begin(), text.end(), [](Text& lhs, Text& rhs)
                {
                    return lhs.box.y < rhs.box.y;
                });
            scale = Size(0, 1);
            shift = Point(0, 1);
        }
        else
        {
            sort(text.begin(), text.end(), [](Text& lhs, Text& rhs)
                {
                    return lhs.box.x < rhs.box.x;
                });
            scale = Size(1, 0);
            shift = Point(1, 0);
        }

        vector<Text> result;

        for (auto& t : text)
        {
            bool added = false;
            for (auto& l : result)
            {
                int dist = min(l.box.width, l.box.height);
                if (!(l.box - (shift * dist) + (scale * dist * 2) & t.box).empty()
                    && (l.avg_width / t.avg_width < 2 && t.avg_width / l.avg_width < 2)) //TODO: need better guess
                {
                    l.box |= t.box;
                    l.avg_width = (l.fill_area * l.avg_width) + (t.avg_width * t.fill_area);
                    l.fill_area += t.fill_area;
                    l.avg_width /= l.fill_area;
                    added = true;
                    break;
                }
            }

            if (added) continue;
            result.push_back(t);
        }

        vector<bool> to_remove(result.size(), false);
        for (int j = 0; j < result.size(); j++)
        {
            if (to_remove[j]) continue;
            Text& l1 = result[j];
            int dist = min(l1.box.width, l1.box.height);
            for (int i = 0; i < result.size(); i++)
            {
                if (to_remove[i]) continue;
                Text& l2 = result[i];
                if (!(l2.box & (l1.box - (shift * dist) + (scale * dist * 2))).empty()
                    && l1.box != l2.box)
                {
                    l1.box |= l2.box;
                    l1.avg_width = (l1.fill_area * l1.avg_width) + (l2.fill_area * l2.avg_width);
                    l1.fill_area += l2.fill_area;
                    l1.avg_width /= l1.fill_area;
                    to_remove[i] = true;
                    i = -1;
                    continue;
                }
            }
        }

        auto it = result.begin();
        for (int i = 0; i < to_remove.size(); i++)
        {
            if (to_remove[i])
            {
                it = result.erase(it);
            }
            else
            {
                it++;
            }
        }

        return result;
    }

    //TODO: Only works with clean, white text bubble
    void MSERTextDetector::detectBubbleTextLine(const Mat& image, vector<Bubble>& bubbles)
    {
        for (auto& b : bubbles)
        {
            Mat bbl(image, b.box);
            Mat bbl_clone = bbl.clone();
            extractBubble(bbl_clone, { b }, 255, -b.box.tl());

            vector<Text> bubble_text = detectTextLine(bbl_clone);
            for (auto& t : bubble_text)
            {
                b.text.push_back(t);
            }

            //for (auto& t : bubble_text)
            //{
            //    rectangle(bbl, t.box, Scalar(180), 3);
            //}
        }
        //showImage(image);
    }
}