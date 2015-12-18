#include <vector>
#include <cv.h>
#include <highgui.h>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace std;
static const char* trackbarR = "R threshould";
static const char* trackbarM = "M threshould";
Mat* imgIx;
Mat* imgIy;
Mat  img;
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
				A += pow(Ix.at<short>(cvPoint(col - 1 + i, row - 1 + j)), 2);

			for (int i = 0; i < apertureSize; i++)
			for (int j = 0; j < apertureSize; j++)
				B += Ix.at<short>(cvPoint(col - 1 + i, row - 1 + j))*Iy.at<short>(cvPoint(col - 1 + i, row - 1 + j));

			C = B;

			for (int i = 0; i < apertureSize; i++)
			for (int j = 0; j < apertureSize; j++)
				D += pow(Iy.at<short>(cvPoint(col - 1 + i, row - 1 + j)), 2);

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
			{
				dst.at<Vec3b>(cvPoint(col, row))[0] = 255;
				dst.at<Vec3b>(cvPoint(col, row))[1] = 0;
				dst.at<Vec3b>(cvPoint(col, row))[2] = 0;
			}
			else
			{
				dst.at<Vec3b>(cvPoint(col, row))[0] *= 0.3;
				dst.at<Vec3b>(cvPoint(col, row))[1] *= 0.3;
				dst.at<Vec3b>(cvPoint(col, row))[2] *= 0.3;
			}
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
	imgHarris = img.clone();
	harrisCorner(*imgIx, *imgIy, imgHarris, 0.06, valueM*1000, valueR*10000000, 3);
	imshow("imgMax", imgMax);
	imshow("imgMin", imgMin);
	imshow("HarrisA", imgHarris);
}

void on_trackM(int pos)
{
	imgHarris = img.clone();
	harrisCorner(*imgIx, *imgIy, imgHarris, 0.06, valueM*1000, valueR *10000000, 3);
	imshow("imgMax", imgMax);
	imshow("imgMin", imgMin);
	imshow("HarrisA", imgHarris);
}

int main(int argc, char** argv[])
{
	img = imread("6.png");
	imgHarris = img.clone();
	imgIx = new Mat(cvSize(img.rows,img.cols),CV_16SC1);
	imgIy = new Mat(cvSize(img.rows, img.cols), CV_16SC1);
	Mat imgBinarized = img.clone();
	cvtColor(imgBinarized, imgBinarized, CV_BGR2GRAY);
	imgMin = imgBinarized.clone();
	imgMax = imgBinarized.clone();

	Sobel(imgBinarized, *imgIx, imgIx->depth(), 1, 0);
	Sobel(imgBinarized, *imgIy, imgIx->depth(), 0, 1);

	harrisCorner(*imgIx, *imgIy, imgHarris, 0.06, valueM*1000, valueR * 10000000, 3);

	imshow("original", img);
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