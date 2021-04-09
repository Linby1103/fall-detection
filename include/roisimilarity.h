/*******************************************************************
 *  Copyright(c) 2000-2013 Company HY
 *  All rights reserved.
 *
 *  文件名称:
 *  简要描述:
 *
 *  当前版本:2.0
 *  作者:
 *  日期:
 *  说明:
 *
 *  取代版本:1.0
 *  作者:
 *  日期:
 *  说明:
 ******************************************************************/

#ifndef DETECTOR_FEATURE_MATCH_H
#define DETECTOR_FEATURE_MATCH_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

float best_match(cv::Mat& templateimg,cv::Mat& queryimg);


/*************************************************
Function:       aHash
Description:    计算图像相似度
Calls:
Input:          matSrc1、matSrc2
Output:
Return:         （double）相似度值 0-1
Others:
*************************************************/
float aHash(cv::Mat matSrc1, cv::Mat matSrc2);

double getMSSIM(const cv::Mat& i1, const cv::Mat& i2);
#endif //DETECTOR_FEATURE_MATCH_H
