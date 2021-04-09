#!/bin/sh
staticlibdir=/mnt/workspace/code/ncnn-master/build/install/lib
staticlibdir2dest= /home/workdir/code/HI3559Av100/nnie_sdk/package/mpp/sample/svp/multi-core/zs_demo_close/lib
sharelibdir=/mnt/workspace/code/ncnn_yolov5/hi_build/lib
sharelibdir2dest= /home/workdir/code/HI3559Av100/nnie_sdk/package/mpp/sample/svp/multi-core/zs_demo_close/opencv/lib

cd /home/workdir/code/HI3559Av100/ncnn-master/build-himix100-gcc-linux/
#rm ./* -rf

#cmake -D NCNN_VULKAN=OFF -D CMAKE_BUILD_TYPE=Release -D NCNN_DISABLE_RTTI=OFF -D CMAKE_TOOLCHAIN_FILE=../toolchains/himix100.toolchain.cmake CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions" ..

make clean
make 
make install
cp ./install/lib/libncnn.a  ${staticlibdir2dest}/ -f

echo "Compile ncnn done!"

cd /home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build/build
make clean
make

cp ../lib/libhi_detector.so ${sharelibdir2dest}/ -f
echo "Compile yolov5 done!"

cd /home/workdir/code/HI3559Av100/nnie_sdk/package/mpp/sample/svp/multi-core/zs_demo_close/
make clean
make

echo "Done!"
