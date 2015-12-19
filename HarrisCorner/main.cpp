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
Mat* imgRThreshold;
Mat* imgRLocalMax;
Mat imgFusion;
Mat imgMax;
Mat imgMin;
int valueR = 100;
int valueL = 50;

void quadraticEquation(float &res1, float &res2, float a, float b, float c)
{//计算一元二次方程的两个根
	res1 = (-b + sqrt(pow(b, 2) - 4 * a*c)) / (2 * a);
	res2 = (-b - sqrt(pow(b, 2) - 4 * a*c)) / (2 * a);
}

void harrisCorner(Mat &Ix, Mat &Iy, Mat &dstRThreshold, Mat &dstRLocalMax, Mat &dstFusion, int k, int threshouldL, long long threshouldR, int apertureSize)
{
	Mat RValue(cvSize(Ix.cols,Ix.rows),CV_32FC1);
	int border = apertureSize / 2;//图像边界
	int row;//行
	int col;//列
	try
	{
		for (row = border; row < Ix.rows - border; row++)
		{
			for (col = border; col < Ix.cols - border; col++)
			{	

				float A, B, C, D;//M矩阵中的四个值
				A = B = C = D = 0;

				for (int i = 0; i < apertureSize; i++)//计算A=窗口中Ix*Ix的和
				for (int j = 0; j < apertureSize; j++)
					A += pow(Ix.at<short>(cvPoint(col - border + i, row - border + j)), 2);

				for (int i = 0; i < apertureSize; i++)//计算B=窗口中Ix*Iy的和
				for (int j = 0; j < apertureSize; j++)
					B += Ix.at<short>(cvPoint(col - border + i, row - border + j))*Iy.at<short>(cvPoint(col - border + i, row - border + j));

				C = B;//C=B

				for (int i = 0; i < apertureSize; i++)//计算D=窗口中Iy*Iy的和
				for (int j = 0; j < apertureSize; j++)
					D += pow(Iy.at<short>(cvPoint(col - border + i, row - border + j)), 2);

				float det = A*D - B*C;//计算矩阵的det
				float trace = A + D;//计算矩阵的trace
				float R = det - k*trace*trace;//计算R值
				RValue.at<float>(cvPoint(col, row)) = R;
				float o1 = 0;//lambda1
				float o2 = 0;//lambda2

				quadraticEquation(o1, o2, 1, -trace, det);//通过矩阵解出两个特征值lambda1,lambda2
				//设置lambdaMax图
				if (o1 > threshouldL || o2 > threshouldL)//如果lambda1,lambda2中较大的一个大于threshold,说明某个方向变化比另一个方向大
					imgMax.at<uchar>(cvPoint(col, row)) = 255;//该点显示为白色
				else
					imgMax.at<uchar>(cvPoint(col, row)) = 0;

				//设置lambdaMin图
				if (o1 > threshouldL && o2 > threshouldL)//如果lambda1，lambda2都大于threshold说明可能为角点
				{
					imgMin.at<uchar>(cvPoint(col, row)) = 255;//该点显示为白色
				
				}
				else
				{
					imgMin.at<uchar>(cvPoint(col, row)) = 0;
				}

				if (R >= threshouldR)//设置R图，和混合图像
				{
					dstRThreshold.at<uchar>(cvPoint(col, row)) = 255;//如果大于threshold，该点为白色
				}
				else
				{
					dstRThreshold.at<uchar>(cvPoint(col, row)) = 0;//如果R小于threshould，该点为黑色
				}
			}
		}

		//计算Local Max
		for (row = border; row < Ix.rows - border-apertureSize; row+=apertureSize)
		{
			for (col = border; col < Ix.cols - border - apertureSize; col += apertureSize)
			{
				int maxI = 0;
				int maxJ = 0;
				float R = RValue.at<float>(cvPoint(col, row));
				for (int i = 0; i < apertureSize;i++)
				for (int j = 0; j < apertureSize; j++)
				{
					if (RValue.at<float>(cvPoint(col + i, row + j))>R)
					{//找出aperature*aperture中极大值点及其R值
						maxI = i;
						maxJ = j;
						R = RValue.at<float>(cvPoint(col + i, row + j));
					}
				}
				//如果该极大值点的R值大于Threshold说明可能是角点
				if (R>threshouldR)
				{
					for (int i = 0; i < apertureSize;i++)
					for (int j = 0; j < apertureSize; j++)
					{
						if (i == maxI&&j == maxJ)
						{
							dstRLocalMax.at<uchar>(cvPoint(col + i, row + j)) = 255;//LocalMax图像该点变白色
							dstFusion.at<Vec3b>(cvPoint(col+i, row+j))[0] = 255;//混合时如果R大于threshold改点为青色
							dstFusion.at<Vec3b>(cvPoint(col+i, row+j))[1] = 255;
							dstFusion.at<Vec3b>(cvPoint(col+i, row+j))[2] = 0;
						}
						else
						{
							dstRLocalMax.at<uchar>(cvPoint(col + i, row + j)) = 0;//LocalMax图像该点变黑色
							dstFusion.at<Vec3b>(cvPoint(col + i, row + j))[0] *= 0.3;//混合时R<threshold改点亮度变暗
							dstFusion.at<Vec3b>(cvPoint(col + i, row + j))[1] *= 0.3;
							dstFusion.at<Vec3b>(cvPoint(col + i, row + j))[2] *= 0.3;
						}						
					}
				}
				else
				{
					for (int i = 0; i < apertureSize; i++)
					for (int j = 0; j < apertureSize; j++)
					{
							dstRLocalMax.at<uchar>(cvPoint(col+i, row+j)) = 0;//LocalMax图像该点变黑色
							dstFusion.at<Vec3b>(cvPoint(col + i, row + j))[0] *= 0.3;//混合时R<threshold改点亮度变暗
							dstFusion.at<Vec3b>(cvPoint(col + i, row + j))[1] *= 0.3;
							dstFusion.at<Vec3b>(cvPoint(col + i, row + j))[2] *= 0.3;
					}
				}
			}
		}
	}
	catch (Exception ex)
	{
		throw ex;
	}
}

void showImage()
{//显示并保存到本地各个图像
	imshow("imgMax", imgMax);
	imwrite("lambdaMax.png", imgMax);
	imshow("imgMin", imgMin);
	imwrite("lambdaMin.png", imgMin);
	imshow("HarrisA", imgFusion);
	imwrite("Harris.png", imgFusion);
	imshow("R>Threshold", *imgRThreshold);
	imwrite("RThreshold.png", *imgRThreshold);
	imshow("R Local Max", *imgRLocalMax);
	imwrite("RLocalMax.png", *imgRLocalMax);
}

void on_trackR(int pos)
{
	imgFusion = img.clone();//重新生成背景
	harrisCorner(*imgIx, *imgIy, *imgRThreshold,*imgRLocalMax, imgFusion, paramK, valueL * 1000, valueR * 10000000, apertureSize);//Harris corner
	showImage();
}

void on_trackL(int pos)
{
	imgFusion = img.clone(); // 重新生成背景
	harrisCorner(*imgIx, *imgIy, *imgRThreshold, *imgRLocalMax, imgFusion, paramK, valueL * 1000, valueR * 10000000, apertureSize);//Harris corner
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
		imageName = argv[1];//第一个参数为图片文件名字
		paramK = atof(argv[2]);//第二个参数为k值
		apertureSize = atoi(argv[3]);//第三个参数为aperature_size
	}

	img = imread(imageName);//打开图像
	imgFusion = img.clone();//复制图像作为Fusion的背景
	imgIx = new Mat(cvSize(img.cols, img.rows), CV_16SC1);
	imgIy = new Mat(cvSize(img.cols, img.rows), CV_16SC1);
	Mat imgBinarized = img.clone();//新建灰度图像
	cvtColor(imgBinarized, imgBinarized, CV_BGR2GRAY);//转为灰度图像
	imgMin = imgBinarized.clone();//复制作为lambdaMin图像基础
	imgMax = imgBinarized.clone();//复制作为lambdaMax图像基础
	imgRThreshold = new Mat(cvSize(img.cols, img.rows), CV_8UC1);//新建imgR即R>threshould的图像
	imgRLocalMax = new Mat(cvSize(img.cols, img.rows), CV_8UC1);//新建imgR即R>threshould的图像

	Sobel(imgBinarized, *imgIx, imgIx->depth(), 1, 0);//Sobel算子计算Ix
	Sobel(imgBinarized, *imgIy, imgIy->depth(), 0, 1);//Sobel算子计算Iy

	harrisCorner(*imgIx, *imgIy, *imgRThreshold, *imgRLocalMax, imgFusion, paramK, valueL * 1000, valueR * 10000000, apertureSize);//Harris corner检测算法

	showImage();//显示图像

	cvCreateTrackbar(//新建拖动条改变R的threshold
		trackbarR,//const char* trackbarName,  
		"HarrisA",//const char* windowName,  
		&valueR,//int* value,  
		210,//int count,  
		on_trackR);//CvTrackbarCallback onChange   
	cvCreateTrackbar(//新建拖动条改变lambda的threshold
		trackbarM,//const char* trackbarName,  
		"HarrisA",//const char* windowName,  
		&valueL,//int* value,  
		100,//int count,  
		on_trackL);//CvTrackbarCallback onChange   

	cvWaitKey();
	return 0;
}