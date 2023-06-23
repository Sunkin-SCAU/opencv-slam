#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iterator>

using namespace std;
using namespace cv;

//建立圖像金字塔
//將原始圖像一級級縮小并依次存在mvImagePyramid里
void ComputePyramid(cv::Mat image)
{// 計算n個level尺度的圖片
    int nlevels = 8;
    int EDGE_THRESHOLD = 16;
    std::vector<cv::Mat> mvImagePyramid;
    mvImagePyramid.resize(nlevels);
    std::vector<float> mvScaleFactor(nlevels,1.0f);		    ///<每层图像的缩放因子
    std::vector<float> mvInvScaleFactor(nlevels,1.0f);        ///<以及每层缩放因子的倒数
    int scaleFactor = 2;


    for(int i=1; i<nlevels; i++)
    {
        //其实就是这样累乘计算得出来的
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
    }
    for (float value : mvScaleFactor) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
    }

    for (float value : mvInvScaleFactor) {
        std::cout << value << " ";
    }
    std::cout << std::endl;


    for (int level = 0; level < nlevels; ++level)
    {
        //获取本层图像的缩放系数
        float scale = mvInvScaleFactor[level];
        //计算本层图像的像素尺寸大小
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        //全尺寸图像。包括无效图像区域的大小。将图像进行“补边”，EDGE_THRESHOLD区域外的图像不进行FAST角点检测
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        // 定义了两个变量：temp是扩展了边界的图像，masktemp并未使用
        Mat temp(wholeSize, image.type()), masktemp;
        // mvImagePyramid 刚开始时是个空的vector<Mat>
        // 把图像金字塔该图层的图像指针mvImagePyramid指向temp的中间部分（这里为浅拷贝，内存相同）
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
//        mvImagePyramid[level] = temp;

        // Compute the resized image
        //计算第0层以上resize后的图像
        if( level != 0 )
        {
            //将上一层金字塔图像根据设定sz缩放到当前层级
            resize(mvImagePyramid[level-1],	//输入图像
                   mvImagePyramid[level], 	//输出图像
                   sz, 						//输出图像的尺寸
                   0, 						//水平方向上的缩放系数，留0表示自动计算
                   0,  						//垂直方向上的缩放系数，留0表示自动计算
                   cv::INTER_LINEAR);		//图像缩放的差值算法类型，这里的是线性插值算法

            // //!  原代码mvImagePyramid 并未扩充，上面resize应该改为如下
            // resize(image,	                //输入图像
            // 	   mvImagePyramid[level], 	//输出图像
            // 	   sz, 						//输出图像的尺寸
            // 	   0, 						//水平方向上的缩放系数，留0表示自动计算
            // 	   0,  						//垂直方向上的缩放系数，留0表示自动计算
            // 	   cv::INTER_LINEAR);		//图像缩放的差值算法类型，这里的是线性插值算法

            //把源图像拷贝到目的图像的中央，四面填充指定的像素。图片如果已经拷贝到中间，只填充边界
            //这样做是为了能够正确提取边界的FAST角点
            //EDGE_THRESHOLD指的这个边界的宽度，由于这个边界之外的像素不是原图像素而是算法生成出来的，所以不能够在EDGE_THRESHOLD之外提取特征点
            copyMakeBorder(mvImagePyramid[level], 					//源图像
                           temp, 									//目标图像（此时其实就已经有大了一圈的尺寸了）
                           EDGE_THRESHOLD, EDGE_THRESHOLD, 			//top & bottom 需要扩展的border大小
                           EDGE_THRESHOLD, EDGE_THRESHOLD,			//left & right 需要扩展的border大小
                           BORDER_REFLECT_101+BORDER_ISOLATED);     //扩充方式，opencv给出的解释：

            /*Various border types, image boundaries are denoted with '|'
            * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
            * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
            * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
            * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
            * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
            */

            //BORDER_ISOLATED	表示对整个图像进行操作
            // https://docs.opencv.org/3.4.4/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36

        }
        else
        {
            //对于第0层未缩放图像，对其周围进行边界扩展。此时temp就是对原图扩展后的图像，这里的temp与mvImagePyramid[level]是浅拷贝的关系，改变了temp也即是改变了mvImagePyramid[level]
            copyMakeBorder(image,			//这里是原图像
                           temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);
        }
        //这里的temp与mvImagePyramid[level]是浅拷贝的关系，改变了temp也即是改变了mvImagePyramid[level]
        //
        //
    }
        for (int i = 0; i < nlevels; i++) {
        std::string windowName = "Layer " + std::to_string(i);
        namedWindow(windowName, WINDOW_NORMAL);
        imshow(windowName, mvImagePyramid[i]);
        }
    waitKey(0);
}

void gausss(cv::Mat image)
{
    namedWindow("原图像", WINDOW_NORMAL);
    imshow("原图像", image);
    cv::Mat GaussImage;
    GaussianBlur(image,GaussImage,Size(7,7),2,2,BORDER_REFLECT_101);
    namedWindow("修改的图像", WINDOW_NORMAL);
    imshow("修改的图像", GaussImage);
    waitKey(0);
}

int main() {
    Mat image = imread("C:\\Users\\x1998\\Desktop\\1.jpg");
//    gausss(image);
    ComputePyramid(image);
    return 0;
}

