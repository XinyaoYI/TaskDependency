
/**
 From https://docs.opencv.org/3.4.0/d3/dc1/tutorial_basic_linear_transform.html
 Check the webpage for description
 */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <sys/timeb.h>
#include <iostream>
#include <omp.h>
 
using namespace std;
using namespace cv;

double read_timer();
/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

int main( int argc, char** argv )
{
    const int MAX_VALUE = 256;//constants
    const int RED_VALUE = 0;
    const int GREEN_VALUE = 1;
    const int BLUE_VALUE = 2;

    String imageName("../data/3.jpg"); 
    if (argc > 1)
    {
        imageName = argv[1];
    }
    Mat image = imread( imageName );//create image 

    int HistRed[MAX_VALUE] = {0};//create histograms
    int HistBlue[MAX_VALUE] = {0};
    int HistGreen[MAX_VALUE] = {0};

    cout << " Basic Histograms " << endl;
    double elapsed_hist = read_timer();//start timer
    
    #pragma omp parallel 
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        printf("me: %d  thread:  %d \n", thread_id, num_threads);

        int localHistRed[MAX_VALUE] = {0};//create histograms
        int localHistBlue[MAX_VALUE] = {0};
        int localHistGreen[MAX_VALUE] = {0};
        #pragma omp for nowait collapse(2) 
        for( int y = 0; y < image.rows; y++ ) {
            for( int x = 0; x < image.cols; x++ ) {
                Vec3b colorValue = image.at<Vec3b>(Point(x, y));//get pixel
                
                int R = colorValue.val[RED_VALUE];//get RGB values for pixel
                int G = colorValue.val[GREEN_VALUE];
                int B = colorValue.val[BLUE_VALUE];


                localHistRed[R] = localHistRed[R]+1;//update histograms
                localHistGreen[G] = localHistGreen[G]+1;
                localHistBlue[B] = localHistBlue[B]+1;   
            }
        }
        #pragma omp critical
        for(int y = 0; y < MAX_VALUE; y++ ) {
            HistRed[y] += localHistRed[y];
            HistBlue[y] += localHistGreen[y];
            HistGreen[y] += localHistBlue[y];
        }

    }  
    elapsed_hist  = (read_timer() - elapsed_hist);//end timer
/*
    Mat HistPlotR (500, 256, CV_8UC3, Scalar(0, 0, 0));
    Mat HistPlotG (500, 256, CV_8UC3, Scalar(0, 0, 0));
    Mat HistPlotB (500, 256, CV_8UC3, Scalar(0, 0, 0));

    for (int i = 0; i < 256; i=i+2)//produce the plots for the historgrams
    {
        line(HistPlotR, Point(i, 500), Point(i, 500-HistRed[i]), Scalar(0, 0, 255),1,8,0);
        line(HistPlotG, Point(i, 500), Point(i, 500-HistGreen[i]), Scalar(0, 255, 0),1,8,0);
        line(HistPlotB, Point(i, 500), Point(i, 500-HistBlue[i]), Scalar(255, 0, 0),1,8,0);
    }

    namedWindow("Original Image", WINDOW_AUTOSIZE);//setup all the windows
    imshow("Original Image", image);

    namedWindow("Red Histogram");
    namedWindow("Green Histogram");
    namedWindow("Blue Histogram");
    imshow("Red Histogram", HistPlotR);
    imshow("Green Histogram", HistPlotG);
    imshow("Blue Histogram", HistPlotB);

*/

    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\t\tRuntime (ms)\t MOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("mm:\t\t\t\t%4f\t%4f\n",  elapsed_hist * 1.0e3, 3*image.cols*image.rows / (1.0e6 *  elapsed_hist));

   // waitKey();
    return 0;
    
}


