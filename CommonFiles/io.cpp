#include "io.h"

#include <windows.h>
#include <commdlg.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include <filesystem>
#include <iostream>
#include <fstream>
#include <map>

#include "psd_sdk/Psd.h"
#include "psd_sdk/PsdPlatform.h"
#include "psd_sdk/PsdMallocAllocator.h"
#include "psd_sdk/PsdExport.h"
#include "psd_sdk/PsdNativeFile.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

namespace mct
{
    vector<wstring> selectImageFromDialog(void)
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

        vector<wstring> files;
        if (GetOpenFileNameW(&ofn))
        {
            wchar_t* strptr = filename;
            fs::path filePath(filename);
            strptr += wcslen(filename) + 1;

            while (*strptr != NULL)
            {
                fs::path fileName(strptr);
                fileName = filePath / fileName;
                files.push_back(fileName.wstring());
                strptr += wcslen(strptr) + 1;
            }

            if (files.size() < 1)
            {
                files.clear();
                files.push_back(filePath.wstring());
            }
        }
        
        return files;
    }

    map<wstring, Mat> loadImages(const vector<wstring>& filenames)
    {
        map<wstring, Mat> images;

        for (wstring s : filenames)
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
                        size_t size = fs::file_size(e.wstring());
                        vector<uchar> buffer(size);
                        ifstream ifs(e.wstring(), ios::in | ios::binary);
                        ifs.read(reinterpret_cast<char*>(&buffer[0]), size);
                        Mat img = imdecode(buffer, 1);
                        //Mat img = imread(e.wstring()); // Read the file
                        if (img.data != NULL)
                        {
                            images[e.wstring()] = img;
                            cout << "Loaded image: " << e.filename() << endl;
                        }
                    }
                }
            }
            else if (fs::is_regular_file(fs::status(p)))
            {
                size_t size = fs::file_size(p.wstring());
                vector<uchar> buffer(size);
                ifstream ifs(p.wstring(), ios::in | ios::binary);
                ifs.read(reinterpret_cast<char*>(&buffer[0]), size);
                Mat img = imdecode(buffer, 1);
                //Mat img = imread(p.wstring()); // Read the file
                if (img.data != NULL)
                {
                    images[p.wstring()] = img;
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

    void saveImage(const Mat& image, wstring filename)
    {
        fs::path p(filename);
        p.replace_filename(L"MCT-" + p.filename().wstring());
        p.replace_extension(".png");
        vector<uchar> buffer;
        imencode(".png", image, buffer);
        ofstream ofs(p.wstring(), ios::out | ios::binary);
        ofs.write(reinterpret_cast<char*>(&buffer[0]), buffer.size());
        cout << "Saved image: " << p.filename() << endl;
    }

    //int writePSD(const Mat& img_original, const Mat& img_mask, wstring filename)
    //{
    //    PSD_USING_NAMESPACE;

    //    MallocAllocator allocator;
    //    NativeFile file(&allocator);

    //    fs::path filePath(filename);
    //    filePath.replace_extension(".psd");

    //    if (!file.OpenWrite(filePath.wstring().c_str()))
    //    {
    //        OutputDebugStringA("Cannot open file.\n");
    //        return 1;
    //    }

    //    ExportDocument* document;
    //    vector<Mat> ch_img;
    //    split(img_original, ch_img);
    //    if (ch_img.size() == 1) //TODO: Support other export format (RGB, 16u, 32f, etc)
    //    {
    //        document = CreateExportDocument(&allocator, img_original.cols, img_original.rows, 8u, exportColorMode::GRAYSCALE);
    //        {
    //            const unsigned int orig = AddLayer(document, &allocator, "Original");
    //            const unsigned int mask = AddLayer(document, &allocator, "Clean Mask");

    //            UpdateLayer(document, &allocator, orig, exportChannel::GRAY, 0, 0, img_original.cols, img_original.rows, (uchar*)ch_img[0].data, compressionType::RLE);

    //            vector<Mat> ch_mask;
    //            split(img_mask, ch_mask);
    //            UpdateLayer(document, &allocator, mask, exportChannel::GRAY, 0, 0, img_mask.cols, img_mask.rows, (uchar*)ch_mask[0].data, compressionType::RLE);
    //            UpdateLayer(document, &allocator, mask, exportChannel::ALPHA, 0, 0, img_mask.cols, img_mask.rows, (uchar*)ch_mask[1].data, compressionType::RLE);

    //            cout << "Writing to file: " << filePath.filename() << endl;
    //            WriteDocument(document, &allocator, &file);

    //            DestroyExportDocument(document, &allocator);
    //        }
    //    }
    //    else if (ch_img.size() == 3)
    //    {
    //        document = CreateExportDocument(&allocator, img_original.cols, img_original.rows, 8u, exportColorMode::RGB);
    //        {
    //            const unsigned int orig = AddLayer(document, &allocator, "Original");
    //            const unsigned int mask = AddLayer(document, &allocator, "Clean Mask");

    //            UpdateLayer(document, &allocator, orig, exportChannel::RED, 0, 0, img_original.cols, img_original.rows, (uchar*)ch_img[0].data, compressionType::RLE);
    //            UpdateLayer(document, &allocator, orig, exportChannel::GREEN, 0, 0, img_original.cols, img_original.rows, (uchar*)ch_img[1].data, compressionType::RLE);
    //            UpdateLayer(document, &allocator, orig, exportChannel::BLUE, 0, 0, img_original.cols, img_original.rows, (uchar*)ch_img[2].data, compressionType::RLE);

    //            vector<Mat> ch_mask;
    //            split(img_mask, ch_mask);
    //            UpdateLayer(document, &allocator, mask, exportChannel::RED, 0, 0, img_mask.cols, img_mask.rows, (uchar*)ch_mask[0].data, compressionType::RLE);
    //            UpdateLayer(document, &allocator, mask, exportChannel::GREEN, 0, 0, img_mask.cols, img_mask.rows, (uchar*)ch_mask[0].data, compressionType::RLE);
    //            UpdateLayer(document, &allocator, mask, exportChannel::BLUE, 0, 0, img_mask.cols, img_mask.rows, (uchar*)ch_mask[0].data, compressionType::RLE);
    //            UpdateLayer(document, &allocator, mask, exportChannel::ALPHA, 0, 0, img_mask.cols, img_mask.rows, (uchar*)ch_mask[1].data, compressionType::RLE);

    //            cout << "Writing to file: " << filePath.filename() << endl;
    //            WriteDocument(document, &allocator, &file);

    //            DestroyExportDocument(document, &allocator);
    //        }
    //    }

    //    file.Close();

    //    return 0;
    ////}
}