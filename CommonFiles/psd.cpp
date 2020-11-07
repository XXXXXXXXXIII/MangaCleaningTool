#include "psd.h"

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
#include "psd_sdk/PsdExportDocument.h"
#include "psd_sdk/PsdNativeFile.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

namespace mct
{
	psd::ExportDocument* createPsdDocument(const cv::Size& size, int type)
	{
		CV_Assert(type == CV_8UC1 || type == CV_8UC3);

        PSD_USING_NAMESPACE;

        MallocAllocator allocator;

        ExportDocument* document;
		if (type == CV_8UC1)
		{
			document = CreateExportDocument(&allocator, size.width, size.height, 8u, exportColorMode::GRAYSCALE);
		}
		else
		{
			document = CreateExportDocument(&allocator, size.width, size.height, 8u, exportColorMode::RGB);
		}

		return document;
	}

	void addPsdLayer(psd::ExportDocument* doc, cv::Mat& image, std::string name)
	{
		PSD_USING_NAMESPACE;
		CV_Assert(image.cols == doc->width && image.rows == doc->height);
		//CV_Assert(!(doc->colorMode == exportColorMode::GRAYSCALE && image.channels() > 2));
		
		MallocAllocator allocator;
		const unsigned int new_layer = AddLayer(doc, &allocator, name.c_str());

		vector<Mat> img_ch;
		split(image, img_ch);
		if (doc->colorMode == exportColorMode::GRAYSCALE)
		{
			switch (img_ch.size())
			{
			case 1:
				UpdateLayer(doc, &allocator, new_layer, exportChannel::GRAY, 0, 0, image.cols, image.rows, (uchar*)img_ch[0].data, compressionType::RLE);
				break;
			case 2:
				UpdateLayer(doc, &allocator, new_layer, exportChannel::GRAY, 0, 0, image.cols, image.rows, (uchar*)img_ch[0].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::ALPHA, 0, 0, image.cols, image.rows, (uchar*)img_ch[1].data, compressionType::RLE);
				break;
			}
		}
		else if(doc->colorMode == exportColorMode::RGB)
		{
			switch (img_ch.size())
			{
			case 1:
				UpdateLayer(doc, &allocator, new_layer, exportChannel::RED, 0, 0, image.cols, image.rows, (uchar*)img_ch[0].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::GREEN, 0, 0, image.cols, image.rows, (uchar*)img_ch[0].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::BLUE, 0, 0, image.cols, image.rows, (uchar*)img_ch[0].data, compressionType::RLE);
				break;
			case 2:
				UpdateLayer(doc, &allocator, new_layer, exportChannel::RED, 0, 0, image.cols, image.rows, (uchar*)img_ch[0].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::GREEN, 0, 0, image.cols, image.rows, (uchar*)img_ch[0].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::BLUE, 0, 0, image.cols, image.rows, (uchar*)img_ch[0].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::ALPHA, 0, 0, image.cols, image.rows, (uchar*)img_ch[1].data, compressionType::RLE);
				break;
			case 3:
				UpdateLayer(doc, &allocator, new_layer, exportChannel::RED, 0, 0, image.cols, image.rows, (uchar*)img_ch[0].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::GREEN, 0, 0, image.cols, image.rows, (uchar*)img_ch[1].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::BLUE, 0, 0, image.cols, image.rows, (uchar*)img_ch[2].data, compressionType::RLE);
				break;
			case 4:
				UpdateLayer(doc, &allocator, new_layer, exportChannel::RED, 0, 0, image.cols, image.rows, (uchar*)img_ch[0].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::GREEN, 0, 0, image.cols, image.rows, (uchar*)img_ch[1].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::BLUE, 0, 0, image.cols, image.rows, (uchar*)img_ch[2].data, compressionType::RLE);
				UpdateLayer(doc, &allocator, new_layer, exportChannel::ALPHA, 0, 0, image.cols, image.rows, (uchar*)img_ch[3].data, compressionType::RLE);
				break;
			}
		}
	}

	void destroyPsdDocument(psd::ExportDocument* doc)
	{
		PSD_USING_NAMESPACE;
		MallocAllocator allocator;
		DestroyExportDocument(doc, &allocator);
	}
	void savePsdDocument(psd::ExportDocument* doc, std::wstring filename)
	{
		PSD_USING_NAMESPACE;

		MallocAllocator allocator;
		NativeFile file(&allocator);

		fs::path filePath(filename);
		filePath.replace_extension(".psd");

		if (!file.OpenWrite(filePath.wstring().c_str()))
		{
			OutputDebugStringA("Cannot open file.\n");
			return;
		}

		cout << "Writing to file: " << filePath.filename() << endl;
		WriteDocument(doc, &allocator, &file);
		file.Close();
	}
};