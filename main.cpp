#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
 VideoCapture webcam;
    if(!webcam.open(0))
        return 0;
    for(;;)
    {
          Mat frame;
          webcam >> frame;
          if( frame.empty() ) break; // end of video stream
          
          imshow("this is you, smile! :)", frame);
          if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
    }
    return 0;
}