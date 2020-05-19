#include "text.h"
#include "mct.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;


// Code from: https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
namespace mct
{
    void EASTTextDetector::detectBubbleText(const cv::Mat& image, const std::vector<Bubble>& bubbles)
    {

        std::vector<Mat> outs;
        std::vector<String> outNames(2);
        outNames[0] = "feature_fusion/Conv_7/Sigmoid";
        outNames[1] = "feature_fusion/concat_3";

        Mat blob;

        for (auto& b : bubbles)
        {
            Mat bbl(image, b.box);
            Mat bbl_clone = bbl.clone();
            extractBubble(bbl_clone, { b }, 255, -b.box.tl());

            blobFromImage(bbl_clone, blob, 1.0, Size(320, 320), Scalar(123.68, 116.78, 103.94), true, false);
            detector.setInput(blob);
            detector.forward(outs, outNames);

            Mat scores = outs[0];
            Mat geometry = outs[1];

            // Decode predicted bounding boxes.
            std::vector<RotatedRect> boxes;
            std::vector<float> confidences;
            decodeBoundingBoxes(scores, geometry, 0.5, boxes, confidences);

            // Apply non-maximum suppression procedure.
            std::vector<int> indices;
            NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

            Point2f ratio((float)bbl.cols / 320, (float)bbl.rows / 320);

            // Render text.
            for (size_t i = 0; i < indices.size(); ++i)
            {
                RotatedRect& box = boxes[indices[i]];

                Point2f vertices[4];
                box.points(vertices);

                for (int j = 0; j < 4; ++j)
                {
                    vertices[j].x *= ratio.x;
                    vertices[j].y *= ratio.y;
                }

                for (int j = 0; j < 4; ++j)
                    line(bbl, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 1);
            }

        }
        showImage(image);
    }

    void EASTTextDetector::decodeBoundingBoxes(const Mat& scores, const Mat& geometry, float scoreThresh,
        std::vector<RotatedRect>& detections, std::vector<float>& confidences)
    {
        detections.clear();
        CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
        CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
        CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

        const int height = scores.size[2];
        const int width = scores.size[3];
        for (int y = 0; y < height; ++y)
        {
            const float* scoresData = scores.ptr<float>(0, 0, y);
            const float* x0_data = geometry.ptr<float>(0, 0, y);
            const float* x1_data = geometry.ptr<float>(0, 1, y);
            const float* x2_data = geometry.ptr<float>(0, 2, y);
            const float* x3_data = geometry.ptr<float>(0, 3, y);
            const float* anglesData = geometry.ptr<float>(0, 4, y);
            for (int x = 0; x < width; ++x)
            {
                float score = scoresData[x];
                if (score < scoreThresh)
                    continue;

                // Decode a prediction.
                // Multiple by 4 because feature maps are 4 time less than input image.
                float offsetX = x * 4.0f, offsetY = y * 4.0f;
                float angle = anglesData[x];
                float cosA = std::cos(angle);
                float sinA = std::sin(angle);
                float h = x0_data[x] + x2_data[x];
                float w = x1_data[x] + x3_data[x];

                Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                    offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
                Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
                Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
                RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
                detections.push_back(r);
                confidences.push_back(score);
            }
        }
    }

    void EASTTextDetector::fourPointsTransform(const Mat& frame, Point2f vertices[4], Mat& result)
    {
        const Size outputSize = Size(100, 32);

        Point2f targetVertices[4] = { Point(0, outputSize.height - 1),
                                      Point(0, 0), Point(outputSize.width - 1, 0),
                                      Point(outputSize.width - 1, outputSize.height - 1),
        };
        Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);

        warpPerspective(frame, result, rotationMatrix, outputSize);
    }

    void EASTTextDetector::decodeText(const Mat& scores, std::string& text)
    {
        static const std::string alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
        Mat scoresMat = scores.reshape(1, scores.size[0]);

        std::vector<char> elements;
        elements.reserve(scores.size[0]);

        for (int rowIndex = 0; rowIndex < scoresMat.rows; ++rowIndex)
        {
            Point p;
            minMaxLoc(scoresMat.row(rowIndex), 0, 0, 0, &p);
            if (p.x > 0 && static_cast<size_t>(p.x) <= alphabet.size())
            {
                elements.push_back(alphabet[p.x - 1]);
            }
            else
            {
                elements.push_back('-');
            }
        }

        if (elements.size() > 0 && elements[0] != '-')
            text += elements[0];

        for (size_t elementIndex = 1; elementIndex < elements.size(); ++elementIndex)
        {
            if (elementIndex > 0 && elements[elementIndex] != '-' &&
                elements[elementIndex - 1] != elements[elementIndex])
            {
                text += elements[elementIndex];
            }
        }
    }
}