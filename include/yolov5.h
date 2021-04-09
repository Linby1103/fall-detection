//
// Created by libin on 2021/3/22.
//

#ifndef NCNN_YOLOV5_YOLOV5_H
#define NCNN_YOLOV5_YOLOV5_H

#include "layer.h"
#include "net.h"

using namespace ncnn;

class YoloV5Focus : public Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const;

};

DEFINE_LAYER_CREATOR(YoloV5Focus)

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

//  interface

#endif //NCNN_YOLOV5_YOLOV5_H
