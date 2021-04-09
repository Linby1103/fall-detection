#指定CMake编译最低要求版本
CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
#给项目命名
PROJECT(detector)
#打印调试信息
MESSAGE(STATUS "Project: ${PROJECT_NAME}")
SET( CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -fno-rtti -fno-exceptions")
#CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions
#set（OpenCV_DIR "/media/libin/办公/Workspace/nnie/nnie_caffe_env/nnie_tools/opencv-3.4.0/build/“）
#SET(OpenCV_DIR /media/libin/办公/Workspace/HI3559AV100/workspace/install/opencv-3.4.0/output_hi3559/)
SET(CMAKE_NCNN_DIR /home/workdir/code/HI3559Av100/ncnn-master/build-himix100-gcc-linux/install/)

set(CMAKE_C_COMPILER /opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-gcc)
set(CMAKE_CXX_COMPILER /opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++)


#收集c/c++文件并赋值给变量SRC_LIST_CPP  ${PROJECT_SOURCE_DIR}代表区当前项目录

FILE(GLOB SRC_LIST_CPP ${PROJECT_SOURCE_DIR}/../src/*.cpp)
FILE(GLOB SRC_LIST_C ${PROJET_SOURCE_DIR}/../src/*.c)
#指定头文件目录
include_directories(${PROJECT_SOURCE_DIR}/../include ${CMAKE_NCNN_DIR}/include/ncnn  /home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build/output_hi3559/include)


#link_directories(/media/libin/办公/Workspace/HI3559AV100/workspace/install/opencv-3.4.0/build/lib ${CMAKE_NCNN_DIR}/lib/)
#ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST_CPP} ${SRC_LIST_C})
#target_link_libraries(${PROJECT_NAME}  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_dnn  -lopencv_ml -fopenmp -lncnn  -lpthread -lm)

#指定生成库文件的目录
#SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#去变量SRC_LIST_CPP 与SRC_LIST_C 指定生成libmyprint 动态库   默认生成静态库  SHARED指定生成库类型为动态库
ADD_LIBRARY(falldetector SHARED ${SRC_LIST_CPP} ${SRC_LIST_C})

# gcc TestLoadshareLib.c -o main -I /usr/local/include -L /usr/local/lib -lAnimal_detector -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_video
