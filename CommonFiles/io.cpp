#include "io.h"

#include <windows.h>
#include <commdlg.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include <filesystem>
#include <iostream>
#include <fstream>
#include <map>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

namespace mct
{
    vector<string> selectImageFromDialog(void)
    {
        wchar_t filename[2560] = {'\0'};

        OPENFILENAME ofn;
        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = NULL;  // If you have a window to center over, put its HANDLE here
        ofn.lpstrFilter = L"Image File\0*.png; *.jpg; *.jpeg\0";
        ofn.lpstrFile = filename;
        ofn.nMaxFile = 2560;
        ofn.lpstrTitle = L"Select image";
        ofn.Flags = OFN_CREATEPROMPT | OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY | OFN_EXPLORER | OFN_ALLOWMULTISELECT;

        vector<string> files;
        if (GetOpenFileNameW(&ofn))
        {
            wchar_t* strptr = filename;
            fs::path filePath(filename);
            strptr += wcslen(filename) + 1;

            while (*strptr != NULL)
            {
                fs::path fileName(strptr);
                fileName = filePath / fileName;
                files.push_back(fileName.string());
                strptr += wcslen(strptr) + 1;
            }

            if (files.size() < 1)
            {
                files.clear();
                files.push_back(filePath.string());
            }
        }
        
        return files;
    }

    map<string, Mat> loadImages(const vector<string>& filenames)
    {
        map<string, Mat> images;

        for (string s : filenames)
        {
            fs::path p(s);
            if (fs::is_directory(fs::status(p)))
            {
                for (const auto& entry : fs::directory_iterator(p))
                {
                    fs::path e(entry);
                    if (e.extension() == ".jpg"
                        || e.extension() == ".png"
                        || e.extension() == ".jpeg")
                    {
                        Mat img = imread(e.string()); // Read the file
                        if (img.data != NULL)
                        {
                            images[e.string()] = img;
                            cout << "Loaded image: " << e.filename() << endl;
                        }
                    }
                }
            }
            else if (fs::is_regular_file(fs::status(p)))
            {
                Mat img = imread(p.string()); // Read the file
                if (img.data != NULL)
                {
                    images[p.string()] = img;
                    cout << "Loaded image: " << p.filename() << endl;
                }
            }
        }

        return images;
    }
    void saveFrameProperty(std::vector<Frame>& frames)
    {
        fs::path p = fs::current_path();
        p.append(frame_prop_file);
        p.replace_extension(".csv");
        cout << "Saving to " << p << endl;

        ofstream outfile;
        outfile.open(p, ios::app);

        for (auto f : frames)
        {
            outfile << fixed << f.toCSVData() << endl;
            cout << fixed << f.toCSVData() << endl;
        }

        outfile.close();
    }

    void saveBubbleProperty(std::vector<Bubble>& bubbles)
    {
        fs::path p = fs::current_path();
        p.append(bubble_prop_file);
        p.replace_extension(".csv");
        cout << "Saving to " << p << endl;

        ofstream outfile;
        outfile.open(p, ios::app);

        for (auto f : bubbles)
        {
            outfile << fixed << f.toCSVData() << endl;
            cout << fixed << f.toCSVData() << endl;
        }

        outfile.close();
    }

    void saveCutouts(const Mat& image, vector<Bubble>& bubbles, int width, int height)
    {
        vector<Rect> rect(bubbles.size());
        vector<bool> tag(bubbles.size());
        for (int i = 0; i < bubbles.size(); i++)
        {
            rect[i] = bubbles[i].box;
            tag[i] = bubbles[i].is_bubble;
        }

        saveCutouts(image, rect, tag, width, height);
    }


    /*
        Save rects from image as png files, used for training
        Will rescale to 50 x 50 by default
        Prepends "1_" to front of file name by default

        @param width: width of the output image, 0 for no scaling
        @param height: height of the output image, 0 for no scaling
        @param manual_tagging: enable manually tagging image, support 0/1
    */
    void saveCutouts(const Mat& image, vector<Rect>& rect, vector<bool>& tag, int width, int height)
    {
        fs::path p = fs::current_path();
        fs::create_directory(p.append("training_data"));
        p.append("temp");
        //srand(chrono::high_resolution_clock::now().time_since_epoch().count());

        for (int i = 0; i < rect.size(); i++)
        {
            Rect r = rect[i];
            string name = to_string(r.x) + to_string(r.y) + to_string(r.br().x) + to_string(r.br().y);
            //cout << p.string() << endl;
            Mat bbl(image, r);
            Mat dst;
            if (width > 0 && height > 0)
            {
                dst = Mat(Size(width, height), CV_8UC1);
            }
            else
            {
                dst = Mat(bbl.size(), CV_8UC1);
            }
            resize(bbl, dst, dst.size(), 0, 0, 0);

            p.replace_filename(to_string(tag[i]) + "_" + name);
            p.replace_extension(".png");
            imwrite(p.string(), dst);
        }
    }
}