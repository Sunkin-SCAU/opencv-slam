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

//�����D�������
//��ԭʼ�D��һ�����sС�����δ���mvImagePyramid��
void ComputePyramid(cv::Mat image)
{// Ӌ��n��level�߶ȵĈDƬ
    int nlevels = 8;
    int EDGE_THRESHOLD = 16;
    std::vector<cv::Mat> mvImagePyramid;
    mvImagePyramid.resize(nlevels);
    std::vector<float> mvScaleFactor(nlevels,1.0f);		    ///<ÿ��ͼ�����������
    std::vector<float> mvInvScaleFactor(nlevels,1.0f);        ///<�Լ�ÿ���������ӵĵ���
    int scaleFactor = 2;


    for(int i=1; i<nlevels; i++)
    {
        //��ʵ���������۳˼���ó�����
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
        //��ȡ����ͼ�������ϵ��
        float scale = mvInvScaleFactor[level];
        //���㱾��ͼ������سߴ��С
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        //ȫ�ߴ�ͼ�񡣰�����Чͼ������Ĵ�С����ͼ����С����ߡ���EDGE_THRESHOLD�������ͼ�񲻽���FAST�ǵ���
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        // ����������������temp����չ�˱߽��ͼ��masktemp��δʹ��
        Mat temp(wholeSize, image.type()), masktemp;
        // mvImagePyramid �տ�ʼʱ�Ǹ��յ�vector<Mat>
        // ��ͼ���������ͼ���ͼ��ָ��mvImagePyramidָ��temp���м䲿�֣�����Ϊǳ�������ڴ���ͬ��
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
//        mvImagePyramid[level] = temp;

        // Compute the resized image
        //�����0������resize���ͼ��
        if( level != 0 )
        {
            //����һ�������ͼ������趨sz���ŵ���ǰ�㼶
            resize(mvImagePyramid[level-1],	//����ͼ��
                   mvImagePyramid[level], 	//���ͼ��
                   sz, 						//���ͼ��ĳߴ�
                   0, 						//ˮƽ�����ϵ�����ϵ������0��ʾ�Զ�����
                   0,  						//��ֱ�����ϵ�����ϵ������0��ʾ�Զ�����
                   cv::INTER_LINEAR);		//ͼ�����ŵĲ�ֵ�㷨���ͣ�����������Բ�ֵ�㷨

            // //!  ԭ����mvImagePyramid ��δ���䣬����resizeӦ�ø�Ϊ����
            // resize(image,	                //����ͼ��
            // 	   mvImagePyramid[level], 	//���ͼ��
            // 	   sz, 						//���ͼ��ĳߴ�
            // 	   0, 						//ˮƽ�����ϵ�����ϵ������0��ʾ�Զ�����
            // 	   0,  						//��ֱ�����ϵ�����ϵ������0��ʾ�Զ�����
            // 	   cv::INTER_LINEAR);		//ͼ�����ŵĲ�ֵ�㷨���ͣ�����������Բ�ֵ�㷨

            //��Դͼ�񿽱���Ŀ��ͼ������룬�������ָ�������ء�ͼƬ����Ѿ��������м䣬ֻ���߽�
            //��������Ϊ���ܹ���ȷ��ȡ�߽��FAST�ǵ�
            //EDGE_THRESHOLDָ������߽�Ŀ�ȣ���������߽�֮������ز���ԭͼ���ض����㷨���ɳ����ģ����Բ��ܹ���EDGE_THRESHOLD֮����ȡ������
            copyMakeBorder(mvImagePyramid[level], 					//Դͼ��
                           temp, 									//Ŀ��ͼ�񣨴�ʱ��ʵ���Ѿ��д���һȦ�ĳߴ��ˣ�
                           EDGE_THRESHOLD, EDGE_THRESHOLD, 			//top & bottom ��Ҫ��չ��border��С
                           EDGE_THRESHOLD, EDGE_THRESHOLD,			//left & right ��Ҫ��չ��border��С
                           BORDER_REFLECT_101+BORDER_ISOLATED);     //���䷽ʽ��opencv�����Ľ��ͣ�

            /*Various border types, image boundaries are denoted with '|'
            * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
            * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
            * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
            * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
            * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
            */

            //BORDER_ISOLATED	��ʾ������ͼ����в���
            // https://docs.opencv.org/3.4.4/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36

        }
        else
        {
            //���ڵ�0��δ����ͼ�񣬶�����Χ���б߽���չ����ʱtemp���Ƕ�ԭͼ��չ���ͼ�������temp��mvImagePyramid[level]��ǳ�����Ĺ�ϵ���ı���tempҲ���Ǹı���mvImagePyramid[level]
            copyMakeBorder(image,			//������ԭͼ��
                           temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);
        }
        //�����temp��mvImagePyramid[level]��ǳ�����Ĺ�ϵ���ı���tempҲ���Ǹı���mvImagePyramid[level]
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
    namedWindow("ԭͼ��", WINDOW_NORMAL);
    imshow("ԭͼ��", image);
    cv::Mat GaussImage;
    GaussianBlur(image,GaussImage,Size(7,7),2,2,BORDER_REFLECT_101);
    namedWindow("�޸ĵ�ͼ��", WINDOW_NORMAL);
    imshow("�޸ĵ�ͼ��", GaussImage);
    waitKey(0);
}

int main() {
    Mat image = imread("C:\\Users\\x1998\\Desktop\\1.jpg");
//    gausss(image);
    ComputePyramid(image);
    return 0;
}

