# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xinyaoyi/Documents/TaskDependency/serial

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xinyaoyi/Documents/TaskDependency/serial/build

# Include any dependencies generated for this target.
include CMakeFiles/Smoothing.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Smoothing.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Smoothing.dir/flags.make

CMakeFiles/Smoothing.dir/Smoothing.cpp.o: CMakeFiles/Smoothing.dir/flags.make
CMakeFiles/Smoothing.dir/Smoothing.cpp.o: ../Smoothing.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xinyaoyi/Documents/TaskDependency/serial/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Smoothing.dir/Smoothing.cpp.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Smoothing.dir/Smoothing.cpp.o -c /home/xinyaoyi/Documents/TaskDependency/serial/Smoothing.cpp

CMakeFiles/Smoothing.dir/Smoothing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Smoothing.dir/Smoothing.cpp.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xinyaoyi/Documents/TaskDependency/serial/Smoothing.cpp > CMakeFiles/Smoothing.dir/Smoothing.cpp.i

CMakeFiles/Smoothing.dir/Smoothing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Smoothing.dir/Smoothing.cpp.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xinyaoyi/Documents/TaskDependency/serial/Smoothing.cpp -o CMakeFiles/Smoothing.dir/Smoothing.cpp.s

CMakeFiles/Smoothing.dir/Smoothing.cpp.o.requires:

.PHONY : CMakeFiles/Smoothing.dir/Smoothing.cpp.o.requires

CMakeFiles/Smoothing.dir/Smoothing.cpp.o.provides: CMakeFiles/Smoothing.dir/Smoothing.cpp.o.requires
	$(MAKE) -f CMakeFiles/Smoothing.dir/build.make CMakeFiles/Smoothing.dir/Smoothing.cpp.o.provides.build
.PHONY : CMakeFiles/Smoothing.dir/Smoothing.cpp.o.provides

CMakeFiles/Smoothing.dir/Smoothing.cpp.o.provides.build: CMakeFiles/Smoothing.dir/Smoothing.cpp.o


# Object files for target Smoothing
Smoothing_OBJECTS = \
"CMakeFiles/Smoothing.dir/Smoothing.cpp.o"

# External object files for target Smoothing
Smoothing_EXTERNAL_OBJECTS =

Smoothing: CMakeFiles/Smoothing.dir/Smoothing.cpp.o
Smoothing: CMakeFiles/Smoothing.dir/build.make
Smoothing: /usr/local/lib/libopencv_dnn.so.4.4.0
Smoothing: /usr/local/lib/libopencv_gapi.so.4.4.0
Smoothing: /usr/local/lib/libopencv_highgui.so.4.4.0
Smoothing: /usr/local/lib/libopencv_ml.so.4.4.0
Smoothing: /usr/local/lib/libopencv_objdetect.so.4.4.0
Smoothing: /usr/local/lib/libopencv_photo.so.4.4.0
Smoothing: /usr/local/lib/libopencv_stitching.so.4.4.0
Smoothing: /usr/local/lib/libopencv_video.so.4.4.0
Smoothing: /usr/local/lib/libopencv_videoio.so.4.4.0
Smoothing: /usr/local/lib/libopencv_imgcodecs.so.4.4.0
Smoothing: /usr/local/lib/libopencv_calib3d.so.4.4.0
Smoothing: /usr/local/lib/libopencv_features2d.so.4.4.0
Smoothing: /usr/local/lib/libopencv_flann.so.4.4.0
Smoothing: /usr/local/lib/libopencv_imgproc.so.4.4.0
Smoothing: /usr/local/lib/libopencv_core.so.4.4.0
Smoothing: CMakeFiles/Smoothing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xinyaoyi/Documents/TaskDependency/serial/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Smoothing"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Smoothing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Smoothing.dir/build: Smoothing

.PHONY : CMakeFiles/Smoothing.dir/build

CMakeFiles/Smoothing.dir/requires: CMakeFiles/Smoothing.dir/Smoothing.cpp.o.requires

.PHONY : CMakeFiles/Smoothing.dir/requires

CMakeFiles/Smoothing.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Smoothing.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Smoothing.dir/clean

CMakeFiles/Smoothing.dir/depend:
	cd /home/xinyaoyi/Documents/TaskDependency/serial/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xinyaoyi/Documents/TaskDependency/serial /home/xinyaoyi/Documents/TaskDependency/serial /home/xinyaoyi/Documents/TaskDependency/serial/build /home/xinyaoyi/Documents/TaskDependency/serial/build /home/xinyaoyi/Documents/TaskDependency/serial/build/CMakeFiles/Smoothing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Smoothing.dir/depend

