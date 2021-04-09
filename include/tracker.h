//
// Created by xupeng on 2021/1/18.
//
#ifndef NCNNPOSE_DEMO_tracker_H
#define NCNNPOSE_DEMO_tracker_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


struct KeyPoint {
    cv::Point2f p;
    float prob;

};

typedef struct _TrackID_
{
    struct KeyPoint kp;      //每个目标bounding box 的中心坐标
    int ID;                  //每一个目标ID
    int lostcount;           //目标连续丢失的次数
    bool iflost;             //判断当前目标是否丢失

    int newidappearcount;    //计算新目标连续出现的次数
    cv::Rect _rect_;         //目标bounding box
    cv::Mat matchdata;       //目标对应ROI

    double displacement;      //位移
    double slop ;              //斜率
}TrackID;

class tracker {
public:
    tracker();
    int init();

    ~tracker();
    void update_track(const std::vector<std::vector<cv::Rect>> &keypoints,cv::Mat &src);
public:

    std::vector<TrackID> vec_currenttarget;
    std::vector<TrackID> vec_newtarget ;
};


#endif //NCNNPOSE_DEMO_tracker_H
