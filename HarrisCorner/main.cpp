#include <vector>
#include <cv.h>
#include <highgui.h>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace std;
static const char* trackbarR = "R threshould";
static const char* trackbarM = "M threshould";
Mat imgIx;
Mat imgIy;
Mat imgHarris;
Mat imgMax;
Mat imgMin;
int valueR = 24;
int valueM = 50;

void quadraticEquation(double &res1, double &res2, double a, double b, double c)
{
	res1 = (-b + sqrt(pow(b, 2) - 4 * a*c)) / (2 * a);
	res2 = (-b - sqrt(pow(b, 2) - 4 * a*c)) / (2 * a);
}

void harrisCorner(Mat &Ix, Mat &Iy, Mat &dst, int k, int threshould, long long threshouldR, int apertureSize)
{
	Mat RValue();
	for (int row = 1; row < Ix.rows - 1; row++)
	{
		for (int col = 1; col < Ix.cols - 1; col++)
		{

			long long A, B, C, D;
			A = B = C = D = 0;

			for (int i = 0; i < apertureSize; i++)
			for (int j = 0; j < apertureSize; j++)
				A += pow(Ix.at<uchar>(cvPoint(col - 1 + i, row - 1 + j)), 2);

			for (int i = 0; i < apertureSize; i++)
			for (int j = 0; j < apertureSize; j++)
				B += Ix.at<uchar>(cvPoint(col - 1 + i, row - 1 + j))*Iy.at<uchar>(cvPoint(col - 1 + i, row - 1 + j));

			C = B;

			for (int i = 0; i < apertureSize; i++)
			for (int j = 0; j < apertureSize; j++)
				D += pow(Iy.at<uchar>(cvPoint(col - 1 + i, row - 1 + j)), 2);

			if (col == 69)
			{
				int a = 0;
				a++;
			}

			long long det = A*D - B*C;
			long long trace = A + D;
			long long R = det - k*trace*trace;
			double o1 = 0;
			double o2 = 0;

			quadraticEquation(o1, o2, 1, -trace, det);
			if (o1 > threshould || o2 > threshould)
				imgMax.at<uchar>(cvPoint(col, row)) = 255;
			else
				imgMax.at<uchar>(cvPoint(col, row)) = 0;

			if (o1 > threshould && o2 > threshould)
				imgMin.at<uchar>(cvPoint(col, row)) = 255;
			else
				imgMin.at<uchar>(cvPoint(col, row)) = 0;

			if (R >= threshouldR)
				dst.at<uchar>(cvPoint(col, row)) = 255;
			else
				dst.at<uchar>(cvPoint(col, row)) = 0;
		}
	}
}

void convolute3(float* temp, Mat &imgSrc, Mat &imgDst)
{
	for (int row = 1; row < imgSrc.rows - 1; row++)
	{
		for (int col = 1; col < imgSrc.cols - 1; col++)
		{
			uchar pixelDst = 0;
			for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				pixelDst += temp[i * 3 + j] * imgSrc.at<uchar>(cvPoint(col - 1 + j, row - 1 + i));

			imgDst.at<uchar>(cvPoint(col, row)) = abs(pixelDst);
		}
	}
}

void convolute2(float* temp, Mat &imgSrc, Mat &imgDst)
{
	for (int row = 1; row < imgSrc.rows - 1; row++)
	{
		for (int col = 1; col < imgSrc.cols - 1; col++)
		{
			uchar pixelDst = 0;
			for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				pixelDst += temp[i * 2 + j] * imgSrc.at<uchar>(cvPoint(col - 1 + j, row - 1 + i));

			imgDst.at<uchar>(cvPoint(col, row)) = pixelDst;
		}
	}
}

void on_trackR(int pos)
{
	harrisCorner(imgIx, imgIy, imgHarris, 0.06, valueM*1000, valueR*10000000, 3);
	imshow("imgMax", imgMax);
	imshow("imgMin", imgMin);
	imshow("HarrisA", imgHarris);
}

void on_trackM(int pos)
{
	harrisCorner(imgIx, imgIy, imgHarris, 0.06, valueM*1000, valueR *10000000, 3);
	imshow("imgMax", imgMax);
	imshow("imgMin", imgMin);
	imshow("HarrisA", imgHarris);
}

int main(int argc, char** argv[])
{
	Mat img = imread("6.png");
	imshow("original", img);
	imgIx = img.clone();
	imgIy = img.clone();
	float SobelX[9];
	SobelX[0] = -1; SobelX[1] = 0; SobelX[2] = 1;
	SobelX[3] = -2; SobelX[4] = 0; SobelX[5] = 2;
	SobelX[6] = -1; SobelX[7] = 0; SobelX[8] = 1;
	float SobelY[9];
	SobelY[0] = 1; SobelY[1] = 2; SobelY[2] = 1;
	SobelY[3] = 0; SobelY[4] = 0; SobelY[5] = 0;
	SobelY[6] = -1; SobelY[7] = -2; SobelY[8] = -1;
	cvtColor(img, img, CV_BGR2GRAY);
	cvtColor(imgIx, imgIx, CV_BGR2GRAY);
	cvtColor(imgIy, imgIy, CV_BGR2GRAY);
	imgHarris = img.clone();
	imgMin = img.clone();
	imgMax = img.clone();

	Sobel(img, imgIx, img.depth(), 1, 0);
	Sobel(img, imgIy, img.depth(), 0, 1);
	//convolute3(SobelX, img, imgIx);
	//convolute3(SobelY, img, imgIy);


	harrisCorner(imgIx, imgIy, imgHarris, 0.06, valueM*1000, valueR * 10000000, 3);
	imshow("imgMax", imgMax);
	imshow("imgMin", imgMin);
	imshow("HarrisA", imgHarris);
	cvCreateTrackbar(
		trackbarR,//const char* trackbarName,  
		"HarrisA",//const char* windowName,  
		&valueR,//int* value,  
		200,//int count,  
		on_trackR);//CvTrackbarCallback onChange   
	cvCreateTrackbar(
		trackbarM,//const char* trackbarName,  
		"HarrisA",//const char* windowName,  
		&valueM,//int* value,  
		200,//int count,  
		on_trackM);//CvTrackbarCallback onChange   

	cvWaitKey();
	return 0;
}