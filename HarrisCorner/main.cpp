#include <vector>
#include <cv.h>
#include <highgui.h>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace std;
static const char* trackbarR = "R threshould";
static const char* trackbarM = "M threshould";
char *imageName;
float paramK;
int apertureSize;
Mat* imgIx;
Mat* imgIy;
Mat  img;
Mat imgR;
Mat imgFusion;
Mat imgMax;
Mat imgMin;
int valueR = 24;
int valueL = 50;

void quadraticEquation(double &res1, double &res2, double a, double b, double c)
{
	res1 = (-b + sqrt(pow(b, 2) - 4 * a*c)) / (2 * a);
	res2 = (-b - sqrt(pow(b, 2) - 4 * a*c)) / (2 * a);
}

void harrisCorner(Mat &Ix, Mat &Iy, Mat &dstR,Mat &dstFusion, int k, int threshouldL, long long threshouldR, int apertureSize)
{
	Mat RValue();
	int border = apertureSize / 2;
	int row;
	int col;
	try
	{
		for (row = border; row < Ix.rows - border; row++)
		{
			for (col = border; col < Ix.cols - border; col++)
			{

				long long A, B, C, D;
				A = B = C = D = 0;

				for (int i = 0; i < apertureSize; i++)
				for (int j = 0; j < apertureSize; j++)
					A += pow(Ix.at<short>(cvPoint(col - border + i, row - border + j)), 2);

				for (int i = 0; i < apertureSize; i++)
				for (int j = 0; j < apertureSize; j++)
					B += Ix.at<short>(cvPoint(col - border + i, row - border + j))*Iy.at<short>(cvPoint(col - border + i, row - border + j));

				C = B;

				for (int i = 0; i < apertureSize; i++)
				for (int j = 0; j < apertureSize; j++)
					D += pow(Iy.at<short>(cvPoint(col - border + i, row - border + j)), 2);

				long long det = A*D - B*C;
				long long trace = A + D;
				long long R = det - k*trace*trace;
				double o1 = 0;
				double o2 = 0;

				quadraticEquation(o1, o2, 1, -trace, det);
				if (o1 > threshouldL || o2 > threshouldL)
					imgMax.at<uchar>(cvPoint(col, row)) = 255;
				else
					imgMax.at<uchar>(cvPoint(col, row)) = 0;

				if (o1 > threshouldL && o2 > threshouldL)
					imgMin.at<uchar>(cvPoint(col, row)) = 255;
				else
					imgMin.at<uchar>(cvPoint(col, row)) = 0;

				if (R >= threshouldR)
				{
					dstR.at<uchar>(cvPoint(col, row)) = 255;
					dstFusion.at<Vec3b>(cvPoint(col, row))[0] = 255;
					dstFusion.at<Vec3b>(cvPoint(col, row))[1] = 255;
					dstFusion.at<Vec3b>(cvPoint(col, row))[2] = 0;
				}
				else
				{
					dstR.at<uchar>(cvPoint(col, row)) = 0;
					dstFusion.at<Vec3b>(cvPoint(col, row))[0] *= 0.3;
					dstFusion.at<Vec3b>(cvPoint(col, row))[1] *= 0.3;
					dstFusion.at<Vec3b>(cvPoint(col, row))[2] *= 0.3;
				}
			}
		}
	}
	catch (Exception ex)
	{
		throw ex;
	}
}

void convolute3(float* temp, Mat &imgSrc, Mat &imgdstR)
{
	for (int row = 1; row < imgSrc.rows - 1; row++)
	{
		for (int col = 1; col < imgSrc.cols - 1; col++)
		{
			uchar pixeldstR = 0;
			for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				pixeldstR += temp[i * 3 + j] * imgSrc.at<uchar>(cvPoint(col - 1 + j, row - 1 + i));

			imgdstR.at<uchar>(cvPoint(col, row)) = abs(pixeldstR);
		}
	}
}

void convolute2(float* temp, Mat &imgSrc, Mat &imgdstR)
{
	for (int row = 1; row < imgSrc.rows - 1; row++)
	{
		for (int col = 1; col < imgSrc.cols - 1; col++)
		{
			uchar pixeldstR = 0;
			for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				pixeldstR += temp[i * 2 + j] * imgSrc.at<uchar>(cvPoint(col - 1 + j, row - 1 + i));

			imgdstR.at<uchar>(cvPoint(col, row)) = pixeldstR;
		}
	}
}

void showImage()
{
	imshow("imgMax", imgMax);
	imwrite("lambdaMax.png", imgMax);
	imshow("imgMin", imgMin);
	imwrite("lambdaMin.png", imgMin);
	imshow("HarrisA", imgFusion);
	imwrite("Harris.png", imgFusion);
	imshow("R", imgR);
	imwrite("R.png", imgR);
}

void on_trackR(int pos)
{
	imgFusion = img.clone();
	harrisCorner(*imgIx, *imgIy, imgR, imgFusion, paramK, valueL * 1000, valueR * 10000000, apertureSize);
	showImage();
}

void on_trackM(int pos)
{
	imgFusion = img.clone();
	harrisCorner(*imgIx, *imgIy, imgR, imgFusion, paramK, valueL * 1000, valueR * 10000000, apertureSize);
	showImage();
}

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("fatal error invalid input, must be filename.xxx parameter");
		exit(0);
	}
	else
	{
		imageName = argv[1];
		paramK = atof(argv[2]);
		apertureSize = atoi(argv[3]);
	}

	img = imread(imageName);
	imgFusion = img.clone();
	imgIx = new Mat(cvSize(img.rows,img.cols),CV_16SC1);
	imgIy = new Mat(cvSize(img.rows, img.cols), CV_16SC1);
	Mat imgBinarized = img.clone();
	cvtColor(imgBinarized, imgBinarized, CV_BGR2GRAY);
	imgMin = imgBinarized.clone();
	imgMax = imgBinarized.clone();
	imgR = imgBinarized.clone();

	Sobel(imgBinarized, *imgIx, imgIx->depth(), 1, 0);
	Sobel(imgBinarized, *imgIy, imgIx->depth(), 0, 1);

	harrisCorner(*imgIx, *imgIy, imgR, imgFusion, paramK, valueL * 1000, valueR * 10000000, apertureSize);

	showImage();

	cvCreateTrackbar(
		trackbarR,//const char* trackbarName,  
		"HarrisA",//const char* windowName,  
		&valueR,//int* value,  
		200,//int count,  
		on_trackR);//CvTrackbarCallback onChange   
	cvCreateTrackbar(
		trackbarM,//const char* trackbarName,  
		"HarrisA",//const char* windowName,  
		&valueL,//int* value,  
		200,//int count,  
		on_trackM);//CvTrackbarCallback onChange   

	cvWaitKey();
	return 0;
}