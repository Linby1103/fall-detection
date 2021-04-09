#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

float best_match(cv::Mat& templateimg,cv::Mat& queryimg)
{
    cv::Mat result;


    int result_cols = queryimg.cols - templateimg.cols + 1;
    int result_rows = queryimg.rows - templateimg.rows + 1;
    result.create(result_cols, result_rows, CV_32FC1);

    matchTemplate(queryimg, templateimg, result, CV_TM_SQDIFF_NORMED);//这里我们使用的匹配算法是标准平方差匹配 method=CV_TM_SQDIFF_NORMED，数值越小匹配度越好
    normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    double minVal = -1;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::Point matchLoc;
    std::cout << "匹配度：" << minVal << std::endl;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
    std::cout << "匹配度：" << minVal << std::endl;

    matchLoc = minLoc;
    rectangle(queryimg, matchLoc, cv::Point(matchLoc.x + templateimg.cols, matchLoc.y + templateimg.rows), cv::Scalar(0, 255, 0), 2, 8, 0);

    imshow("img", queryimg);
    cv::waitKey(0);

    return 0;
}


float aHash(cv::Mat matSrc1, cv::Mat matSrc2)
{
    cv::Mat matDst1, matDst2;
    cv::resize(matSrc1, matDst1, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
    cv::resize(matSrc2, matDst2, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);

    cv::cvtColor(matDst1, matDst1, CV_BGR2GRAY);
    cv::cvtColor(matDst2, matDst2, CV_BGR2GRAY);

    int iAvg1 = 0, iAvg2 = 0;
    int arr1[64], arr2[64];

    for (int i = 0; i < 8; i++)
    {
        uchar* data1 = matDst1.ptr<uchar>(i);
        uchar* data2 = matDst2.ptr<uchar>(i);

        int tmp = i * 8;

        for (int j = 0; j < 8; j++)
        {
            int tmp1 = tmp + j;

            arr1[tmp1] = data1[j] / 4 * 4;
            arr2[tmp1] = data2[j] / 4 * 4;

            iAvg1 += arr1[tmp1];
            iAvg2 += arr2[tmp1];
        }
    }

    iAvg1 /= 64;
    iAvg2 /= 64;

    for (int i = 0; i < 64; i++)
    {
        arr1[i] = (arr1[i] >= iAvg1) ? 1 : 0;
        arr2[i] = (arr2[i] >= iAvg2) ? 1 : 0;
    }

    int iDiffNum = 0;

    for (int i = 0; i < 64; i++)
        if (arr1[i] == arr2[i])
            ++iDiffNum;
    return (float)(iDiffNum)/64;
}

int  main_t(int argc,char* argv[])
{
    cv::Mat img1=cv::imread("/mnt/tool/dataset_tools_self/video_image_objget/human_tumble/2021-01-19-13-15-560_0.jpg");
    cv::Mat img2=cv::imread("/mnt/tool/dataset_tools_self/video_image_objget/human_tumble/2021-01-19-13-16-160_0.jpg");
    float x=aHash(img1,img2);


    return 0;
}