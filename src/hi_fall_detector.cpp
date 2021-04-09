////
//// Created by libin on 2021/3/22.
////
//
//#include "iostream"
//#include <opencv2/opencv.hpp>
//#include "hi_fall_detector.h"
//int main(int argc,char* argv[])
//{
//
//    const char* param_path="/mnt/workspace/code/ncnn_yolov5/ncnn_model/v0.1/yolov5_best-sim.param";
//    const char* bin_path="/mnt/workspace/code/ncnn_yolov5/ncnn_model/v0.1/yolov5_best-sim.bin";
//    std::string  imagepath="/mnt/workspace/code/ncnn_yolov5/ncnn_model/v0.1/frame0111.jpg";
//    cv::Mat image=cv::imread(imagepath);
//    int iw=image.cols;
//    int ih=image.rows;
//
//    Bbox res_bbox[20];
//    void* detector=creat_detector(param_path,bin_path);
//    detect(detector,image.data,iw,ih,res_bbox);
//    destory_detector(detector);
//
//
//}


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

int main1(int argc,char * argv[])
{
    cv::Mat img=cv::imread("/home/workdir/code/RK3399_M1808/yolov5/model_dir/test_dir/image/inf_human/000356.jpg");
    cv::imshow("demo",img);
    cv::waitKey(0);
    return 0;
}