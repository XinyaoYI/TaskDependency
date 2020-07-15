## Assignment 1 – CSCE569, Spring 2018
------





In this assignment, you will implement two image processing algorithms, Image Histogram and Smoothing. [OpenCV](https://opencv.org/) will be used for reading, writing and displaying images, and for getting and setting pixel values of an image. The `read_timer()` function provided in `mm.c` or `matvec.c` can be used for timing purpose.  The output of your program should include both the time of computation (ms) and FLOPS performance.

#### Histogram calculation
Image histogram provides a graphical representation of the intensity distribution of an image by quantifying the number of pixels for each intensity value considered. The description of image histogram and an OpenCV example for calcualting histogram can be found from [OpenCV Tutorial for Histogram Calculation](https://docs.opencv.org/3.4.0/d8/dbc/tutorial_histogram_calculation.html), and [this C code](http://homepages.inf.ed.ac.uk/rbf/BOOKS/PHILLIPS/cips2edsrc/HIST.C) provide a very concise implementation for the algorithm.

#### Image smoothing
Image smoothing, which also referred to as image blurring is a simple and frequently used image processing operation. Depending the effects we want the smoothing operation to be applied to an image, we can apply different filter (coefficients matrix) to an image using the same algorithms. The description and an OpenCV example can be found from [OpenCV Tutorial for Smoothing Images](https://docs.opencv.org/3.4.0/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html), and [this C code](http://homepages.inf.ed.ac.uk/rbf/BOOKS/PHILLIPS/cips2edsrc/FILTER.C) provide concise code for image filtering algorithms. 

For your implementation, while you can look at the OpenCV's implementation for Histogram calculation and Image smoothing, they are much more complicated that we we need for the purpose of this assignment. It would be much easier you adapt the implmentation in the two C codes to the OpenCV image processing framework.

The methods for reading/writing/displaying an image and for getting/setting a pixel value of an image are pretty simple in OpenCV, which can be found from [this short description with code sample for operating images](https://docs.opencv.org/3.4.0/d5/d98/tutorial_mat_operations.html). For your implementation, the recommendation is to start with [the program for changing the contrast and brightness of an image](https://docs.opencv.org/3.4.0/d3/dc1/tutorial_basic_linear_transform.html), which provides executable codes for those operations. 

## Source Code to Start
[The Assignment_1.zip](../Assignment_1.zip) package contains all the files you need, including mm.c, matvec.c, Histogram.cpp, and Smoothing.cpp, cmake CMakeLists.txt, image data, other sample souce files, an excel file for creating figures, this webpage, and a README.md file. **The Histogram.cpp, Smoothing.cpp and DisplayImage.cpp files for OpenCV image processing are all from the OpenCV tutorial mentioned above and they are for your reference only. For your Histogram and Smoothing implementation, you should start with the ContractBrightness.cpp file and modify the code according to the implementation done in HIST.C and FILTER.C.** README.md provide instructions for how to build each executable.

## [Machine to use for development](../resources/HardwareSoftware.html)

**Your submission should be a single zipped file named LastNameFirstName.zip that include source code files and one report document file named as LastNameFirstNameAssignment1.pdf. The easiest way is to do your implementation in a folder unpacked from Assignment_1.zip. After that, add the report file and re-package the folder and submit it.** The source files contain your implementations, and each file should be invidually compiled to generate executables. The report is max 3-page report that includes: 

1. In your report, compute and report CPU peak performance based on the CPU you are using,  and then compute and report the performance efficiency of each function (mm, mv, histogram and smoothing). Please be noted that your program is sequential and only use one core, so the performance efficiency = program FLOPS / per-core peak. 

1. Report: 40 points. 
 1. Although homework assignments will not be pledged, per se, the submitted solutions must be your work and not copied from other students' assignments or other sources. 
 1. You may not transmit or receive code from anyone in the class in any way--visually (by showing someone your code), electronically (by emailing, posting, or otherwise sending someone your code), verbally (by reading your code to someone), or in any other way.
 1. You may not collaborate with people who are not your classmates, TAs, or instructor in any way. For example, you may not post questions to programming forums. 
 1. Any violations of these rules will be reported to the honor council. Check the syllabus for the late policy and academic conduct. 