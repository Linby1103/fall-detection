// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "yolov5.h"
#include "hi_fall_detector.h"
#include <stdio.h>
#include <vector>
#include "tracker.h"
#include "falldetection.h"
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

//宏值  V5相关参数
#define INPUT_SIZE 416
#define PROB_THRESHOLD  0.45f
#define NMS_THRESHOLD  0.45f
#define MAX_OBJECT_NUM 20

/***********************************fall detector *****************************/
void pose(tracker* pose,FALLDetection *detector ,cv::Mat &img ,std::vector<Object>& vDetectResults)
{
    int xmin=-1,ymin=-1,wbox=-1,hbox=-1;
    std::vector<cv::Rect> keypoints;

    std::vector<cv::Rect> bodybox;
    static int framecounter=0;
    std::vector<std::vector<cv::Rect>> wholekeypoint;
    wholekeypoint.clear();
    bodybox.clear();

    int imgw=img.cols;
    int imgh=img.rows;

    for(std::vector<Object>::iterator it = vDetectResults.begin(); it != vDetectResults.end(); it++)
    {
        keypoints.clear();
        if(it->label ==0 || it->label ==1)
        {
            int tlx=int(it->rect.tl().x);
            int tly=int(it->rect.tl().y);
            int brx=int(it->rect.br().x);
            int bry=int(it->rect.br().y) ;

            if(tlx<0) tlx=0;
            if(tly<0) tly=0;
            if (brx>=imgw) brx=imgw-1;
            if (bry>=imgh) bry=imgh-1;

            cv::Mat humanroi=img(cv::Rect(cv::Point(tlx,tly),cv::Point(brx,bry))).clone();

            cv::Rect rect;
            rect.x=tlx;
            rect.y=tly;
            rect.width=brx-tlx;
            rect.height=bry-tly;

            keypoints.push_back(rect);

            if(keypoints.size()>0)
            {
                wholekeypoint.push_back(keypoints);
            }

        }

        if (it->label ==0){

            int tlx=int(it->rect.tl().x);
            int tly=int(it->rect.tl().y);
            int brx=int(it->rect.br().x);
            int bry=int(it->rect.br().y) ;

            if(tlx<0) tlx=0;
            if(tly<0) tly=0;
            if (brx>=imgw) brx=imgw-1;
            if (bry>=imgh) bry=imgh-1;

            cv::Rect rect;
            rect.x=tlx;
            rect.y=tly;
            rect.width=brx-tlx;
            rect.height=bry-tly;
            bodybox.push_back(rect);
        }
    }

    if(framecounter==0){
        pose->update_track(wholekeypoint, img);
        framecounter=1;
    }
    else{
        pose->update_track(wholekeypoint, img);

//        for(int i=0;i,pose->)
    }
    static int counter=0;
    int x=detector->feature_combination(pose->vec_currenttarget,bodybox);
    if (x<0)
    {
        char imgname[50];
        sprintf(imgname,"./fall/%d_demo.jpg",counter);
        cv::imwrite(imgname,img);
        counter++;
    }

    for(int i=0;i<pose->vec_currenttarget.size();i++)
    {
        if(pose->vec_currenttarget[i].ID==-1)
            continue;
        std::string text = std::to_string(pose->vec_currenttarget[i].ID);
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 2;
        int thickness = 2;
        int baseline;
        //获取文本框的长宽
        cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
        cv::putText(img, text, pose->vec_currenttarget[i].kp.p, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);

        cv::circle(img,pose->vec_currenttarget[i].kp.p,5.5,cv::Scalar (255,0,0),2);
        if(pose->vec_currenttarget[i].displacement>25  && false)
        {

            std::string jpegname="./fall/fall_"+std::to_string(pose->vec_currenttarget[i].displacement)+".jpg";
            cv::imwrite(jpegname,img);
        }
    }

    for(int i=0;i<detector->curtarget.size();i++)
    {
        std::string text = std::to_string(detector->curtarget[i].ID);
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 2;
        int thickness = 2;
        int baseline;
        //获取文本框的长宽
        cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
        cv::putText(img, text, detector->curtarget[i].rect.tl(), font_face, font_scale, cv::Scalar(255, 0, 0), thickness, 8, 0);

    }
}

static void scan_dir(char *dir,std::vector<std::string> &list,int depth)
{
    DIR *p_dir = NULL;
    struct dirent *p_entry = NULL;
    struct stat statbuf;

    if((p_dir = opendir(dir)) == NULL)
    {
        printf("can't open dir.\n");
        return;
    }

    chdir (dir);
    while(NULL != (p_entry = readdir(p_dir))) { // 获取下一级目录信息

        lstat(p_entry->d_name, &statbuf);   // 获取下一级成员属性
        if(S_IFDIR & statbuf.st_mode) {      // 判断下一级成员是否是目录
            if (strcmp(".", p_entry->d_name) == 0 || strcmp("..", p_entry->d_name) == 0)
                continue;

            printf("%*s%s/\n", depth, "", p_entry->d_name);
            char subpath[256];
            strcpy(subpath,dir);
            strcat(subpath,p_entry->d_name);
            strcat(subpath,"/");

            printf("%*s%s\n", depth, "", subpath);  // 输出属性不是目录的成员
            list.push_back(subpath);
//            scan_dir(subpath, list,depth+4); // 扫描下一级目录的内容
        } else {
            char abpath[256];
            strcpy(abpath,dir);
            strcat(abpath,p_entry->d_name);
            list.push_back(abpath);
            printf("%*s%s\n", depth, "", p_entry->d_name);  // 输出属性不是目录的成员

        }
    }
//    chdir(".."); // 回到上级目录
    chdir("/home/workdir/code/HI3559Av100/ncnn_yolov5/build/");
    closedir(p_dir);
}



/***************************************************************************/
int YoloV5Focus::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = w / 2;
    int outh = h / 2;
    int outc = channels * 4;

    top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outc; p++)
    {
        const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                *outptr = *ptr;

                outptr += 1;
                ptr += 2;
            }

            ptr += w;
        }
    }
    return 0;
}


static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}

static int detect_yolov5(ncnn::Net* yolov5,const cv::Mat& bgr, std::vector<Object>& objects,std::string parampath,std::string binpath)
{



    const int target_size = 640;
    const float prob_threshold = 0.5f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov5->create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output", out);

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("417", out);

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("437", out);

        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
   // static const char* class_names[] = {"fall"};
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

//    cv::imwrite("./test.jpg",image);
    cv::imshow("image", image);
   // cv::waitKey(0);
}

void* creat_detector(const char* param_path,const char* bin_path)
{
    if ((strlen(param_path) == 0)  || (strlen(bin_path) == 0))
    {
        return NULL;
    }

    ncnn::Net* yolov5=new ncnn::Net();
    yolov5->opt.use_bf16_storage = true;
    yolov5->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    yolov5->load_param(param_path);
    yolov5->load_model(bin_path);
    return (void*)yolov5;
}



int detect(void* net,uchar* data,int img_w,int img_h,Bbox* res_bbox)
{
    std::vector<Object> objects;
    objects.clear();
    ncnn::Net* detector=(ncnn::Net*) net;
    detector->opt.use_bf16_storage = true;


    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)INPUT_SIZE / w;
        w = INPUT_SIZE;
        h = h * scale;
    }
    else
    {
        scale = (float)INPUT_SIZE / h;
        h = INPUT_SIZE;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = detector->create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output", out);

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, PROB_THRESHOLD, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("417", out);

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, PROB_THRESHOLD, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("437", out);

        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, PROB_THRESHOLD, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESHOLD);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count && i<MAX_OBJECT_NUM; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
        printf("xmin %.3f  ymin %.3f  xmax  %.3f  ymax  %.3f \n",x0,y0,x1,y1);

        //get result
        res_bbox[i].rect.x = x0;
        res_bbox[i].rect.y = y0;
        res_bbox[i].rect.width = x1 - x0;
        res_bbox[i].rect.hight = y1 - y0;
        res_bbox[i].label=objects[i].label;
        res_bbox[i].prob=objects[i].prob;

    }

    return 0;
}

void destory_detector(void* detector)
{
    if(detector!=NULL)
    {
        delete detector;
        detector=NULL;
    }
}

int mainx(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
//    detect_yolov5(m, objects,argv[2],argv[3]);

    draw_objects(m, objects);

    return 0;
}



int main(int argc, char** argv)
{
    ncnn::Net yolov5;
    yolov5.opt.use_vulkan_compute = true;

    FALLDetection extractfeat;
    tracker poseobj = tracker();
    // yolov5.opt.use_bf16_storage = true;

    yolov5.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    // original pretrained model from https://github.com/ultralytics/yolov5
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    yolov5.load_param(argv[2]);
    yolov5.load_model(argv[3]);


    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    cv::Mat frame;
    cv::VideoCapture inputVideo(imagepath);        // Open input
    if ( !inputVideo.isOpened())
    {

        return -1;
    }
    int counter=0;

    while (true)
    {

        inputVideo >> frame;            //读取当前帧
        if (frame.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }
        counter+=1;
        if ((counter%4)!=0)
            continue;
        std::vector<Object> objects;
        detect_yolov5(&yolov5,frame, objects,argv[2],argv[3]);

        pose(&poseobj,&extractfeat,frame,objects);
        draw_objects(frame, objects);
        char imgname[50];
        sprintf((char*)imagepath,"./image/image_%d.jpg",counter);
        cv::imwrite(imagepath,frame);

        if(cv::waitKey(20) == 'q')   //延时20ms,获取用户是否按键的情况，如果按下q，会推出程序
            break;
    }

    inputVideo.release();     //释放摄像头资源
    cv::destroyAllWindows();   //释放全部窗口
    return 0;
}




int main123(int argc,char* argv[])
{
    std::vector<std::string > list,sublist;
    ncnn::Net yolov5;
    yolov5.opt.use_vulkan_compute = true;

    FALLDetection extractfeat;
    tracker poseobj = tracker();
    yolov5.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    yolov5.load_param(argv[2]);
    yolov5.load_model(argv[3]);

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    scan_dir((char* )argv[1],sublist,0);
    printf("LENGTH=%d\n",sublist.size());
    for (int j=0;j<sublist.size();j++)
    {
        if(j%3!=0) continue;
        printf("%s\n",sublist[j].c_str());
        cv::Mat img=cv::imread(sublist[j]);
        if(img.data==NULL) -1;
        std::vector<Object> objects;
        detect_yolov5(&yolov5,img, objects,argv[2],argv[3]);
        pose(&poseobj,&extractfeat,img,objects);
        draw_objects(img, objects);
        if(cv::waitKey(30) == 'q')   //延时20ms,获取用户是否按键的情况，如果按下q，会推出程序
            break;
    }
    cv::destroyAllWindows();   //释放全部窗口
}
