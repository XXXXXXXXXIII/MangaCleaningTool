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
        Mat img_bin = image.clone();
        bitwise_not(image, img_bin);
        threshold(img_bin, img_bin, 32, 255, THRESH_TOZERO);
        bitwise_not(img_bin, img_bin);
        //showImage(img_bin);

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
        //    //cout << l.avg_width << " , " << l.fill_area << endl;
        //}
        //showImage(image);

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

    /*
        TODO: Ensure every returned char is alone
        1. remove overlap
        2. sort ascending
        3. for each component, merge in short directions
        4. if similar to max size, break (max size = max length^2)
    */
    std::vector<Text> MSERTextDetector::detectTextChar(const cv::Mat& image)
    {
        CV_Assert(!image.empty());
        CV_Assert(image.type() == CV_8UC1);

        double min_val, max_val;
        minMaxLoc(image, &min_val, &max_val);
        if (min_val > 100 || max_val < 155) return vector<Text>();
        
        Mat img_bin = image.clone();
        threshold(img_bin, img_bin, -1, 255, THRESH_OTSU);

        vector<Text> text, final_text;
        vector<vector<Point>> point;
        vector<Rect> box;
        mser->detectRegions(img_bin, point, box);

        Mat img_swt = strokeWidthTransform(image);

        //vector<float> score(box.size());
        for (int i = 0; i < box.size(); i++)
        {
            if (box[i].area() > image.size().area() / 2) continue;
            Text t = { box[i], 0, 0 };
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

        sort(text.begin(), text.end(), [](const Text& lhs, const Text& rhs)
            {
                return lhs.box.area() > rhs.box.area();
            });

        double avg_sqr = 0;
        vector<bool> keep(text.size(), true);
        for (int i = 0; i < text.size(); i++)
        {
            if (!keep[i]) continue;
            if (text[i].box.size().aspectRatio() < 0.9
                || text[i].box.size().aspectRatio() > 1.1)
            {
                for (int j = i + 1; j < text.size(); j++)
                {
                    if (!keep[j]) continue;
                    Rect n = text[j].box | text[i].box;
                    if (std::max(n.width, n.height) <= 1.1 * avg_sqr)
                    {
                        text[i].box = n;
                        text[i].avg_width = text[i].avg_width * text[i].fill_area + text[j].avg_width * text[j].fill_area;
                        text[i].fill_area += text[j].fill_area;
                        text[i].avg_width /= text[i].fill_area;
                        keep[j] = false;

                        cout << max(n.width, n.height) << endl;
                    Mat temp = image.clone();
                    rectangle(temp, n, Scalar(180), 2);
                    rectangle(temp, text[i].box, Scalar(0), 1);
                    rectangle(temp, text[j].box, Scalar(100), 1);
                    showImage(temp, "", 1);
                    }


                    if (text[i].box.size().aspectRatio() < 0.9
                        || text[i].box.size().aspectRatio() > 1.1)
                    {
                        break;
                    }
                }
            }
            else
            {
                double temp = (avg_sqr + max(text[i].box.width, text[i].box.height)) / 2;
                if (temp > avg_sqr)
                {
                    avg_sqr = temp;
                    cout << avg_sqr << endl;
                }
            }

            //for (int j = i + 1; j < text.size(); j++)
            //{
            //    if (!keep[j]) continue;
            //    if ((text[j].box & text[i].box) == text[j].box)
            //    {
            //        keep[j] = false;
            //    }
            //}
        }

        for (int i = 0; i < text.size(); i++)
        {
            if (keep[i])
            {
                final_text.push_back(text[i]);
                //cout << text[i].box << endl;
                //rectangle(image, text[i].box, Scalar(180), 2);
            }
        }
        //showImage(image);

        return final_text;
    }

    // Assumes clean, white text bubble
    void MSERTextDetector::detectBubbleTextLine(const Mat& image, vector<Bubble>& bubbles)
    {
        for (auto& b : bubbles)
        {
            Mat bbl(image, b.box);
            Mat bbl_clone = bbl.clone();
            extractBubble(bbl_clone, { b }, 255, -b.box.tl());

            b.text = detectTextLine(bbl_clone);

            //b.text = detectTextChar(bbl_clone);
            for (auto& t : b.text)
            {
                //rectangle(image, b.box, Scalar(0), 2);
                //drawContours(image, vector<vector<Point>>{b.contour}, -1, Scalar(100), 1);
                //rectangle(bbl, t.box, Scalar(180), 2);
            }
            //showImage(bbl_clone);
        }
        //showImage(image);
    }
}