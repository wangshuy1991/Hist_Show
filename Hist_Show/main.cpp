
//#include "immintrin.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

// Boost includes
//#include <filesystem.hpp>
//#include <filesystem/fstream.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <math.h>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
	abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

cv::Point origin;
cv::Rect selection;
cv::Mat ROI;
bool selectRect = false;
bool calculateHist = false;

void onMouse(int event, int x, int y, int flags , void* param);

void Histgram(cv::Mat img, cv::Mat mask);
int main(int argc, char **argv)
{
	//// for image show //////
	cv::Mat img_input;
	cv::Mat img_output;

	cv::namedWindow("output", CV_WINDOW_AUTOSIZE);

	cv::VideoCapture cap;  //MVI_1101  MVI_0986_2  MVI_0993 Shiseido Demo Video shutterstock_v24250754.mov
	cap.open(0);
	//cap.set(CV_CAP_PROP_SETTINGS, 1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);																		//cv::VideoCapture cap("../Resource/videos/ff20170425_135238.mkv");  //ff20170425_134749 ff20170425_135238  24250754  8708572

	if (!cap.isOpened())
	{
		FATAL_STREAM("Failed to open video source");
		return 1;
	}
	else INFO_STREAM("Device or file opened");

	cap >> img_input;
	flip(img_input, img_input, 1);
	//// for image show //////

	cv::setMouseCallback("output", onMouse);

	while (!img_input.empty())
	{
		// Grab a frame
		// Update the frame count
		cap >> img_input;
		flip(img_input, img_input, 1);
		//cap >> img_input;

		if (selection.width > 0 && selection.height > 0) //鼠标左键被按下后，该段语句开始执行
		{                  
			cv::rectangle(img_input, selection, cv::Scalar(0, 0, 255), 1, 8, 0);
		}

		if (calculateHist) {
			cv::Mat(img_input,selection).copyTo(ROI);
			Histgram(ROI, cv::Mat());
		}

		img_input.copyTo(img_output);

		cv::imshow("output", img_output);
		//imwrite("../testimage/out.png", img_output);
		char c = cvWaitKey(30);
		if (c >= 0)
			break;
	}
	cap.release();
	std::exit(0);
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	//cv::Mat* img = (cv::Mat*)param;

	if (selectRect) //鼠标左键被按下后，该段语句开始执行
	{                  //按住左键拖动鼠标的时候，该鼠标响应函数
					   //会被不断的触发，不断计算目标矩形的窗口
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
		//cv::rectangle(*img, selection, cv::Scalar(0,0,255), 1, 8, 0);
	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		calculateHist = false;
		origin = cv::Point(x, y);
		selection = cv::Rect(x, y, 0, 0);
		selectRect = true;
		break;
	case CV_EVENT_LBUTTONUP:
		selectRect = false;//直到鼠标左键抬起，标志着目标区域选择完毕。selectObject被置为false
		if (selection.width > 0 && selection.height > 0)
			calculateHist = true; //当在第一帧用鼠标选定了合适的目标跟踪窗口后，calculateHist的值置为 1
		break;
	}
}


void Histgram(cv::Mat img, cv::Mat mask) {
	/////// hsv color space /////////
	cv::Mat img_hsv;
	cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
	img_hsv.convertTo(img_hsv, CV_8UC3);

	int hhistSize = 180;
	int shistSize = 256;
	/// Set the ranges ( for B,G,R) )
	float hrange[] = { 0, 180 };
	float srange[] = { 0, 256 };
	const float* hhistRange = { hrange };
	const float* shistRange = { srange };
	bool uniform = true; bool accumulate = false;

	std::vector<cv::Mat> hsv_planes;
	split(img_hsv, hsv_planes);
	cv::Mat h_hist, s_hist, v_hist;
	/// Compute the histograms:
	calcHist(&hsv_planes[0], 1, 0, mask, h_hist, 1, &hhistSize, &hhistRange, uniform, accumulate);
	calcHist(&hsv_planes[1], 1, 0, mask, s_hist, 1, &shistSize, &shistRange, uniform, accumulate);
	calcHist(&hsv_planes[2], 1, 0, mask, v_hist, 1, &shistSize, &shistRange, uniform, accumulate);
	/// Normalize the result to [ 0, histImage.rows ]

	// Draw the histograms for B, G and R
	int hist_w = 360; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / shistSize);
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::normalize(h_hist, h_hist, 0, histImage.rows, cv::NORM_MINMAX, -1);// , cv::Mat());
	cv::normalize(s_hist, s_hist, 0, histImage.rows, cv::NORM_MINMAX, -1);// , cv::Mat());
	cv::normalize(v_hist, v_hist, 0, histImage.rows, cv::NORM_MINMAX, -1);// , cv::Mat());
	for (int i = 1; i < hhistSize; i++)
	{
		//std::cout << g_hist.at<float>(i - 1) << std::endl;
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(h_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(h_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
	}
	for (int i = 1; i < shistSize; i++)
	{
		//std::cout << g_hist.at<float>(i - 1) << std::endl;
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(s_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(s_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(v_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(v_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("calcHistS Demo", histImage);
}