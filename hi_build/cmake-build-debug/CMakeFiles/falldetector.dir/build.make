# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/workdir/IDE_tools/clion-2020.3.1/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/workdir/IDE_tools/clion-2020.3.1/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/falldetector.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/falldetector.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/falldetector.dir/flags.make

CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.o: CMakeFiles/falldetector.dir/flags.make
CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.o: /home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.o"
	/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.o -c /home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp

CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.i"
	/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp > CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.i

CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.s"
	/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp -o CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.s

CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.o: CMakeFiles/falldetector.dir/flags.make
CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.o: /home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.o"
	/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.o -c /home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp

CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.i"
	/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp > CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.i

CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.s"
	/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp -o CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.s

# Object files for target falldetector
falldetector_OBJECTS = \
"CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.o" \
"CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.o"

# External object files for target falldetector
falldetector_EXTERNAL_OBJECTS =

libfalldetector.so: CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/hi_fall_detector.cpp.o
libfalldetector.so: CMakeFiles/falldetector.dir/home/workdir/code/HI3559Av100/ncnn_yolov5/src/yolov5.cpp.o
libfalldetector.so: CMakeFiles/falldetector.dir/build.make
libfalldetector.so: CMakeFiles/falldetector.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libfalldetector.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/falldetector.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/falldetector.dir/build: libfalldetector.so

.PHONY : CMakeFiles/falldetector.dir/build

CMakeFiles/falldetector.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/falldetector.dir/cmake_clean.cmake
.PHONY : CMakeFiles/falldetector.dir/clean

CMakeFiles/falldetector.dir/depend:
	cd /home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build /home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build /home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build/cmake-build-debug /home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build/cmake-build-debug /home/workdir/code/HI3559Av100/ncnn_yolov5/hi_build/cmake-build-debug/CMakeFiles/falldetector.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/falldetector.dir/depend

