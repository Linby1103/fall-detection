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
#include <iostream>
#include <malloc.h>
#include "opencv2/opencv.hpp"
#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H
class Fall_Classifier{
public:
    Fall_Classifier();
    int falldetection(std::vector<float>& features,int featdim=2);
    void init();
public:
    cv::Ptr<cv::ml::SVM> model;

};

#endif