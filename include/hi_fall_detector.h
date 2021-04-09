//
// Created by libin on 2021/3/22.
//
#ifdef __cplusplus
extern "C" {
#endif

#ifndef NCNN_YOLOV5_HI_FALL_DETECTOR_H
#define NCNN_YOLOV5_HI_FALL_DETECTOR_H
#define COORDINATE_NUM 4

typedef struct _rect
{
    float x;
    float y;
    float width;
    float hight;
} Rect;
typedef struct _bbox
{
    Rect rect;
    int label;
    float prob;
} Bbox;

void* creat_detector(const char* param_path,const char* bin_path);
void destory_detector(void* detector);

int detect(void *net,uchar* data,int img_w,int img_h,Bbox* res_bbox);
//int detect(void *net,uchar* data,int img_w,int img_h);

#ifdef __cplusplus
}
#endif

#endif //NCNN_YOLOV5_HI_FALL_DETECTOR_H