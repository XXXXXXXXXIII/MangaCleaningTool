#pragma once

#ifndef __MCT_PSD_H__
#define __MCT_PSD_H__

#include <opencv2/core.hpp>
#include "psd_sdk/Psd.h"
#include "psd_sdk/PsdPlatform.h"
#include "psd_sdk/PsdExport.h"

namespace mct
{
	psd::ExportDocument* createPsdDocument(const cv::Size& size, int type = CV_8UC1);
	void addPsdLayer(psd::ExportDocument* doc, cv::Mat& image, std::string name);
	void destroyPsdDocument(psd::ExportDocument* doc);
	void savePsdDocument(psd::ExportDocument* doc, std::wstring filename);

};

#endif