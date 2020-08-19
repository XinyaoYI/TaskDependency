 /**
 * file Smoothing.cpp
 * brief Sample code for simple filters
 * author OpenCV team
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <sys/timeb.h>
#include <omp.h>
#include <string>

using namespace std;
using namespace cv;

/// Global Variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;


char window_name[] = "Smoothing Demo";

/// Function headers
int display_caption( const char* caption );
int display_dst( int delay, Mat image );
int smooth(Mat &src, Mat &dest);
double read_timer();

const int MAX = 255;
const int RED_VALUE = 0;
const int GREEN_VALUE = 1;
const int BLUE_VALUE = 2;

/**
 * function main
 */
int main( int argc, char ** argv )
{
    cout << " Basic Filter" << endl;
    string pathForFolder= "";
    if (argc < 2) {
        fprintf(stderr, "give a folder containing images for smoothing EX: \"../data\" \n");
        exit(1);
    } else {
        pathForFolder = argv[1];

    }

    cv::String path(pathForFolder+"/*.jpg");//path to files
    vector<cv::String> fn;//data strcuture for files

    Mat *image[36];//declare matrices
    //Mat new_image;
    Mat *image_to_write[36]; /* make this just a pointer */

    glob(path, fn, true);//preload

    double elapsed_smooth = read_timer();//start timer
    
    #pragma omp parallel num_threads(8)
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        printf("me: %d  thread:  %d \n", thread_id, num_threads);
        #pragma omp single
        {

                image[0] = new Mat();
                *image[0] = imread(fn[0]);

                if(image[0]->empty())
                {
                    cout << " file read error" << endl;
                }
                image_to_write[0] = new Mat( Mat::zeros( image[0]->size(), image[0]->type() ) );
        }
        
        for(int k=0; k<fn.size(); k++)
        {
            if (k<(fn.size()-1)) {
               #pragma omp single nowait
               {
                  //
                  image[k+1] = new Mat();
                  *image[k+1] = imread(fn[k+1]);
                
                  if(image[k+1]->empty())
                  {
                    cout << " file read error" << endl;
                  }
                  image_to_write[k+1] = new Mat( Mat::zeros( image[k+1]->size(), image[k+1]->type() ) );
               }
            }
            for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
            {
                smooth(*image[k], *image_to_write[k]);//implicit barrier because of for inside smooth
                #pragma omp single
                {
                    *image[k] = *image_to_write[k];
                }
            }
            
            #pragma omp single nowait
            {
                stringstream ss;
                ss << k;
                imwrite("../output/" + ss.str() + "out.jpg", *image_to_write[k]);
                delete image_to_write[k];
            }
        }
    }
   
    elapsed_smooth  = (read_timer() - elapsed_smooth);//end timer

    

    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\t\tRuntime (ms)\t MOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("mm:\t\t\t\t%4f\t\n",  elapsed_smooth * 1.0e3);

    return 0;
}

int smooth(Mat &src, Mat &dest)
{
    
    //my choice of filter
    int lpf_filter_16[3][3] =
    {{1, 2, 1},
     {2, 4, 2},
     {1, 2, 1}};

    int sum = 0, a = 0, b = 0, filterType = 16;
    #pragma omp for schedule(guided)
    for( int y = 1; y < src.rows-1; y++ ) {
        for( int x = 1; x < src.cols-1; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                sum = 0;
                for(a=-1; a<2; a++){
                    for(b=-1; b<2; b++){
                       sum = sum + (src.at<Vec3b>(Point(x+b, y+a)).val[c] * lpf_filter_16[a+1][b+1]);
                    }
                }
                sum = sum/filterType;
                if(sum < 0)   sum = 0;
                if(sum > MAX) sum = MAX;
                dest.at<Vec3b>(Point(x, y)).val[c] = sum;
            }
        }
    }
    return 0;
}

int display_dst( int delay , Mat image)
{
    imshow( "New Image", image);
    int c = waitKey ( delay );
    if( c >= 0 ) { return -1; }
    return 0;
}
/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

