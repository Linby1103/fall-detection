##指定CMake编译最低要求版本
#CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
##给项目命名
#PROJECT(detector)
##打印调试信息
#MESSAGE(STATUS "Project: ${PROJECT_NAME}")
#SET( CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
#
##set（OpenCV_DIR "/media/libin/办公/Workspace/nnie/nnie_caffe_env/nnie_tools/opencv-3.4.0/build/“）
#SET(CMAKE_NCNN_DIR /home/workdir/code/HI3559Av100/ncnn-master/build/install)
#find_package(OpenCV 3 REQUIRED)
#MESSAGE(STATUS "OpenCV library status:")
#MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
#MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
#MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#
##收集c/c++文件并赋值给变量SRC_LIST_CPP  ${PROJECT_SOURCE_DIR}代表区当前项目录
#
#FILE(GLOB SRC_LIST_CPP ${PROJECT_SOURCE_DIR}/src/*.cpp)
#FILE(GLOB SRC_LIST_C ${PROJECT_SOURCE_DIR}/src/*.c)
##指定头文件目录
#INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include ${CMAKE_NCNN_DIR}/include/ncnn)
#
#ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST_CPP} ${SRC_LIST_C})
#
#link_directories(${OpenCV_DIR}/lib ${PROJECT_NAME}/lib)
#target_link_libraries(${PROJECT_NAME} -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_dnn  -lopencv_ml -fopenmp -lncnn  -lpthread -lm)
#
##指定生成库文件的目录
#SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
##去变量SRC_LIST_CPP 与SRC_LIST_C 指定生成libmyprint 动态库   默认生成静态库  SHARED指定生成库类型为动态库
#ADD_LIBRARY(falldetector SHARED ${SRC_LIST_CPP} ${SRC_LIST_C})
#
## gcc TestLoadshareLib.c -o main -I /usr/local/include -L /usr/local/lib -lAnimal_detector -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_video




#指定CMake编译最低要求版本
CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
#给项目命名
PROJECT(detector)
#打印调试信息
MESSAGE(STATUS "Project: ${PROJECT_NAME}")
SET( CMAKE_CXX_FLAGS "-std=c++11 -fopenmp ")
SET(CMAKE_NCNN_DIR /home/workdir/code/HI3559Av100/ncnn-master/build/install/)
find_package(OpenCV 3 REQUIRED)
set(CMAKE_BUILD_TYPE Debug)
#收集c/c++文件并赋值给变量SRC_LIST_CPP  ${PROJECT_SOURCE_DIR}代表区当前项目录

FILE(GLOB SRC_LIST_CPP ${PROJECT_SOURCE_DIR}/src/*.cpp)
FILE(GLOB SRC_LIST_C ${PROJET_SOURCE_DIR}/src/*.c)
#指定头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include ${CMAKE_NCNN_DIR}/include/ncnn)

link_directories(${OpenCV_DIR}/lib ${CMAKE_NCNN_DIR}/lib/)
ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST_CPP} ${SRC_LIST_C})
target_link_libraries(${PROJECT_NAME}  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_dnn  -lopencv_ml -fopenmp -lncnn  -lpthread -lm)

#指定生成库文件的目录
#SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#去变量SRC_LIST_CPP 与SRC_LIST_C 指定生成libmyprint 动态库   默认生成静态库  SHARED指定生成库类型为动态库
#ADD_LIBRARY(falldetector SHARED ${SRC_LIST_CPP} ${SRC_LIST_C})

# gcc TestLoadshareLib.c -o main -I /usr/local/include -L /usr/local/lib -lAnimal_detector -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_video
