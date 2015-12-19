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
{//����һԪ���η��̵�������
	res1 = (-b + sqrt(pow(b, 2) - 4 * a*c)) / (2 * a);
	res2 = (-b - sqrt(pow(b, 2) - 4 * a*c)) / (2 * a);
}

void harrisCorner(Mat &Ix, Mat &Iy, Mat &dstRThreshold, Mat &dstRLocalMax, Mat &dstFusion, int k, int threshouldL, long long threshouldR, int apertureSize)
{
	Mat RValue(cvSize(Ix.cols,Ix.rows),CV_32FC1);
	int border = apertureSize / 2;//ͼ��߽�
	int row;//��
	int col;//��
	try
	{
		for (row = border; row < Ix.rows - border; row++)
		{
			for (col = border; col < Ix.cols - border; col++)
			{	

				float A, B, C, D;//M�����е��ĸ�ֵ
				A = B = C = D = 0;

				for (int i = 0; i < apertureSize; i++)//����A=������Ix*Ix�ĺ�
				for (int j = 0; j < apertureSize; j++)
					A += pow(Ix.at<short>(cvPoint(col - border + i, row - border + j)), 2);

				for (int i = 0; i < apertureSize; i++)//����B=������Ix*Iy�ĺ�
				for (int j = 0; j < apertureSize; j++)
					B += Ix.at<short>(cvPoint(col - border + i, row - border + j))*Iy.at<short>(cvPoint(col - border + i, row - border + j));

				C = B;//C=B

				for (int i = 0; i < apertureSize; i++)//����D=������Iy*Iy�ĺ�
				for (int j = 0; j < apertureSize; j++)
					D += pow(Iy.at<short>(cvPoint(col - border + i, row - border + j)), 2);

				float det = A*D - B*C;//��������det
				float trace = A + D;//��������trace
				float R = det - k*trace*trace;//����Rֵ
				RValue.at<float>(cvPoint(col, row)) = R;
				float o1 = 0;//lambda1
				float o2 = 0;//lambda2

				quadraticEquation(o1, o2, 1, -trace, det);//ͨ����������������ֵlambda1,lambda2
				//����lambdaMaxͼ
				if (o1 > threshouldL || o2 > threshouldL)//���lambda1,lambda2�нϴ��һ������threshold,˵��ĳ������仯����һ�������
					imgMax.at<uchar>(cvPoint(col, row)) = 255;//�õ���ʾΪ��ɫ
				else
					imgMax.at<uchar>(cvPoint(col, row)) = 0;

				//����lambdaMinͼ
				if (o1 > threshouldL && o2 > threshouldL)//���lambda1��lambda2������threshold˵������Ϊ�ǵ�
				{
					imgMin.at<uchar>(cvPoint(col, row)) = 255;//�õ���ʾΪ��ɫ
				
				}
				else
				{
					imgMin.at<uchar>(cvPoint(col, row)) = 0;
				}

				if (R >= threshouldR)//����Rͼ���ͻ��ͼ��
				{
					dstRThreshold.at<uchar>(cvPoint(col, row)) = 255;//�������threshold���õ�Ϊ��ɫ
				}
				else
				{
					dstRThreshold.at<uchar>(cvPoint(col, row)) = 0;//���RС��threshould���õ�Ϊ��ɫ
				}
			}
		}

		//����Local Max
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
					{//�ҳ�aperature*aperture�м���ֵ�㼰��Rֵ
						maxI = i;
						maxJ = j;
						R = RValue.at<float>(cvPoint(col + i, row + j));
					}
				}
				//����ü���ֵ���Rֵ����Threshold˵�������ǽǵ�
				if (R>threshouldR)
				{
					for (int i = 0; i < apertureSize;i++)
					for (int j = 0; j < apertureSize; j++)
					{
						if (i == maxI&&j == maxJ)
						{
							dstRLocalMax.at<uchar>(cvPoint(col + i, row + j)) = 255;//LocalMaxͼ��õ���ɫ
							dstFusion.at<Vec3b>(cvPoint(col+i, row+j))[0] = 255;//���ʱ���R����threshold�ĵ�Ϊ��ɫ
							dstFusion.at<Vec3b>(cvPoint(col+i, row+j))[1] = 255;
							dstFusion.at<Vec3b>(cvPoint(col+i, row+j))[2] = 0;
						}
						else
						{
							dstRLocalMax.at<uchar>(cvPoint(col + i, row + j)) = 0;//LocalMaxͼ��õ���ɫ
							dstFusion.at<Vec3b>(cvPoint(col + i, row + j))[0] *= 0.3;//���ʱR<threshold�ĵ����ȱ䰵
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
							dstRLocalMax.at<uchar>(cvPoint(col+i, row+j)) = 0;//LocalMaxͼ��õ���ɫ
							dstFusion.at<Vec3b>(cvPoint(col + i, row + j))[0] *= 0.3;//���ʱR<threshold�ĵ����ȱ䰵
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
{//��ʾ�����浽���ظ���ͼ��
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
	imgFusion = img.clone();//�������ɱ���
	harrisCorner(*imgIx, *imgIy, *imgRThreshold,*imgRLocalMax, imgFusion, paramK, valueL * 1000, valueR * 10000000, apertureSize);//Harris corner
	showImage();
}

void on_trackL(int pos)
{
	imgFusion = img.clone(); // �������ɱ���
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
		imageName = argv[1];//��һ������ΪͼƬ�ļ�����
		paramK = atof(argv[2]);//�ڶ�������Ϊkֵ
		apertureSize = atoi(argv[3]);//����������Ϊaperature_size
	}

	img = imread(imageName);//��ͼ��
	imgFusion = img.clone();//����ͼ����ΪFusion�ı���
	imgIx = new Mat(cvSize(img.cols, img.rows), CV_16SC1);
	imgIy = new Mat(cvSize(img.cols, img.rows), CV_16SC1);
	Mat imgBinarized = img.clone();//�½��Ҷ�ͼ��
	cvtColor(imgBinarized, imgBinarized, CV_BGR2GRAY);//תΪ�Ҷ�ͼ��
	imgMin = imgBinarized.clone();//������ΪlambdaMinͼ�����
	imgMax = imgBinarized.clone();//������ΪlambdaMaxͼ�����
	imgRThreshold = new Mat(cvSize(img.cols, img.rows), CV_8UC1);//�½�imgR��R>threshould��ͼ��
	imgRLocalMax = new Mat(cvSize(img.cols, img.rows), CV_8UC1);//�½�imgR��R>threshould��ͼ��

	Sobel(imgBinarized, *imgIx, imgIx->depth(), 1, 0);//Sobel���Ӽ���Ix
	Sobel(imgBinarized, *imgIy, imgIy->depth(), 0, 1);//Sobel���Ӽ���Iy

	harrisCorner(*imgIx, *imgIy, *imgRThreshold, *imgRLocalMax, imgFusion, paramK, valueL * 1000, valueR * 10000000, apertureSize);//Harris corner����㷨

	showImage();//��ʾͼ��

	cvCreateTrackbar(//�½��϶����ı�R��threshold
		trackbarR,//const char* trackbarName,  
		"HarrisA",//const char* windowName,  
		&valueR,//int* value,  
		210,//int count,  
		on_trackR);//CvTrackbarCallback onChange   
	cvCreateTrackbar(//�½��϶����ı�lambda��threshold
		trackbarM,//const char* trackbarName,  
		"HarrisA",//const char* windowName,  
		&valueL,//int* value,  
		100,//int count,  
		on_trackL);//CvTrackbarCallback onChange   

	cvWaitKey();
	return 0;
}