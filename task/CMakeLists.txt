cmake_minimum_required(VERSION 2.8)
project( OpenCVExample )

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package( OpenCV REQUIRED )

add_executable( Smoothing Smoothing.cpp )
target_link_libraries( Smoothing ${OpenCV_LIBS} )

add_executable( jacobi jacobi.cpp )
target_link_libraries( jacobi ${OpenCV_LIBS} )

add_executable( Histogram Histogram.cpp )
target_link_libraries( Histogram ${OpenCV_LIBS} )
