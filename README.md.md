#                     ncnn上实现基于yolov5的跌倒检测算法

## 1、依赖环境

ubuntu18.04

cmake 3.1(>=3.1)

torch 1.7

ncnn

## 2、开发工具

clion 20

pycharm 20

## 3、编译

cd ncnn_yolov5/build

cmake ..

make

在 ncnn_yolov5/build 下生成detector可执行文件

注：该项目依赖与ncnn，也可以将跌倒判断部分代码移植到其他的目标检测代码中，只需要单独封装接口即可,ncnn编译参考:https://mp.csdn.net/editor/html/115037435

## 4、运行

./detector 视频文件  .param 文件  .bin文件

## 5、描述

跌倒检测基于视频序列，首先用yolo检测人体的外接box,然后对每一个box进行目标跟踪，再对跟踪box进行特征提取，这里只是提取了5个维度上的特征，用svm作二分类器。

## 6、目录结构

├── build  编译目录下面的fall.txt，test.txt是手动提取的跌倒状态特征,可用svm进行 拟合
├── hi_build ├──build  编译海思开平台动态库文件的目录

​                              └── CMakeLists.txt 指定海思交叉编译工具连的CMakeList.txt

├── include 头文件
├── lib 以依赖的库文件
├── ncnn_model  存放ncnn的参数文件
├── src             cpp文件

├── CMakeLists.txt            指定x86的CMakeList.txt

├── SVM            用于训练而分类分类器



​                           