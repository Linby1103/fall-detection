//
// Created by libin on 2021/3/8.
//

#ifndef DETECTOR_FALLDETECTION_H
#define DETECTOR_FALLDETECTION_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tracker.h"
#include "classifier.h"

struct detectobj{
public:
    detectobj();
public:
    cv::Rect rect;
    int ID;
    double headheight;
    double xdet;
    double ydet;
    double aspect;
    double displacement;
    double bottom;
    float iou;
    int appearcounter;
};
class FALLDetection{
public:
    FALLDetection(){init();}
    int feature_combination(std::vector<TrackID> &vec_currenttarget,std::vector<cv::Rect>& boxs);
private:
    bool point_in_rect(cv::Rect &rect, cv::Point2f &pt);
    void init();
public:
    std::vector<detectobj> pretarget;
    std::vector<detectobj> curtarget;
    Fall_Classifier svmclassifier;

};

#endif //DETECTOR_FALLDETECTION_H
