#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
using namespace cv;
using namespace std;

//////////////BASIC_MAT////////////////////////
    
    //make ROI (Region of Interest) , J change then I change
    Mat I=imread(argv[1],1);
    Mat J(I,Rect(10,20,100,200)); //J=I crop x=10,y=20,width=100,height=200
    
    // ROI=I
    Mat I=imread(argv[1],1);
    Mat J(I); //J=I
        //or 
    J=I;    
    
    //make a copy of I
    Mat J = I.clone();
        //or
    Mat J;
    I.copyTo(J);

    //make a mask
    dst=Scalar::all(0); //dst fill with black
    src.copyTo(dst,binary_mask); 
    	//binary_mask is binary Mat (just black(0) and white(255))
    	//black = hide, white = show
    imshow("123",dst);

    //make a blank mat
    Mat I;
    I.create( src.size(), src.type() );

//////////////OPEN_WEBCAM_AND_VIDEO/////////////

        VideoCapture webcam;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!webcam.open(0))
        return 0;
    for(;;)
    {
          Mat frame;
          webcam >> frame;
          if( frame.empty() ) break; // end of video stream
     	  ...
          imshow("this is you, smile! :)", frame);
          if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
    }
/////////////MEASURE_CALC_TIME////////////////
double t = (double)getTickCount();
// do something ...
t = ((double)getTickCount() - t)/getTickFrequency();
cout << "Times passed in seconds: " << t << endl;    

/////////////PIXEL_ACCESS///////////////////

printf("%d\n",imagex.at<uchar>(96,267) ); //y,x
printf("%d\n",imagex.at<Vec3b>(96,267)[0] ); //y,x ,0,1,2:BGR

        ///////MAKE_CONTRAST_AND_OFFSET////
for( int y = 0; y < image.rows; y++ )
{ 
    for( int x = 0; x < image.cols; x++ )
    { 
        for( int c = 0; c < 3; c++ )
        {
        new_image.at<Vec3b>(y,x)[c] =
        saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
        //values out of range or not integers (if α is float), we use
        //saturate_cast to make sure the values are valid
        }
    }
}

    //eficient method
CV_Assert(I.depth() == CV_8U);
const int channels = I.channels();
switch(channels)
{
case 1:
{
    MatIterator_<uchar> it, end;
    for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
    *it = table[*it];
    break;
}
case 3:
{
    MatIterator_<Vec3b> it, end;
    for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
    {
        (*it)[0] = table[(*it)[0]];
        (*it)[1] = table[(*it)[1]];
        (*it)[2] = table[(*it)[2]];
    }
}
}

////////////CONVERT_RGB2GRAY//////////////////

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
int main(int argc, char const *argv[])
{
    if (argc!=2)
    {
        printf("enter an URL\n");
        return -1;
    }
    Mat image=imread(argv[1],1);
    if (!image.data)
    {
        printf("can't open image\n");
        return -1;
    }
    imshow("display image",image);
    Mat imagex;
    cvtColor(image,imagex,CV_BGR2GRAY);
    imshow("display imagex",imagex);
    imwrite("a.jpg",imagex);
    waitKey(0);
    return 0;
}

////////////SPLIT_3CHANNEL_OF_IMAGE_INTO_MAT/////////////////

vector<Mat> bgr_planes;
split( src, bgr_planes );

////////////MATRIX_CONVOLUTION//////////////

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
using namespace cv;
using namespace std;
int main(int argc, char const *argv[])
{
    Mat Hx=(Mat_<char>(3,3) << -1,0,1,
                                 -2,0,2,
                                 -1,0,1);
    Mat Hy=(Mat_<char>(3,3) << -1,-2,-1,
                                  0,0,0,
                                  1,2,1);
       
    Mat I=imread(argv[1],0);
    
    Mat Jx;
    Mat Jy;
    filter2D(I,Jx,I.depth(),Hx);
    filter2D(I,Jy,I.depth(),Hy);

    Mat J=imread(argv[1],0);
    
    for (int i = 0; i < I.rows; i++)
    {       
        for (int j = 0; j < I.cols; j++)
        {
            J.at<uchar>(i,j)=(int)sqrt(Jx.at<uchar>(i,j)*Jx.at<uchar>(i,j)+Jy.at<uchar>(i,j)*Jy.at<uchar>(i,j));     
        }
    }

    imshow("display",I);   
    imshow("displayX",Jx);
    imshow("displayY",Jy);
    imshow("displayJ",J);
    waitKey(0);
    return 0;
}

/////////////ADD_TWO_IMAGE/////////////////////

addWeighted( src1, alpha, src2, beta, gamma, dst);
    //dst = α · src1 + β · src2 + γ
    ex: addWeighted(I,0.3,J,0.7,0.0,K);
    

////////////DRAW_SHAPE/////////////////////////

    ///////init black or white Mat///////
     Mat J=Mat::ones(2,5,CV_8UC1)*255; //white
     Mat J=Mat::zeros(2,5,CV_8UC1); //black

     ////basic/////
line(I,Point(1,1),Point(100,200),Scalar(0,255,0),-1 /*thickness*/,8/*lineType*/);
ellipse(I,Point(100,50),Size(20,40),0 /*angle rotate*/,45/*start arc*/,360/*end arc*/,Scalar(255,0,0),-1/*thickness*/,8/*lineType*/);
circle(I,Point(100,200),30,Scalar(0,0,255),-1);
rectangle(I,Point(100,200),Point(50,100),Scalar( 0, 255, 255 ),-1,8);

    //////polygon//////////
Point rook_points[1][20];
rook_points[0][0] = Point( w/4.0, 7*w/8.0 );
rook_points[0][1] = Point( 3*w/4.0, 7*w/8.0 );
rook_points[0][2] = Point( 3*w/4.0, 13*w/16.0 );
...
const Point* ppt[1] = { rook_points[0] };
int npt[] = { 20 };
fillPoly( I,ppt,npt,1/*number of polygon*/,Scalar( 255, 255, 255 ),8/*lineType*/);

//////////////////TRACKBAR////////////////////////////////
void on_trackbar( int, void * )
{
    //CODE WHEN GRAB TRACKBAR
}
void main()
{
    /// Initialize values
    int alpha_slider = 0;
    /// Create Windows
    namedWindow("Linear Blend", 1);
    /// Create Trackbars
    char TrackbarName[50];
    int alpha_slider_max=100;
    createTrackbar( TrackbarName, "Linear Blend", &alpha_slider,alpha_slider_max,on_trackbar );
    /// Show some stuff
    on_trackbar( alpha_slider, 0 );
}

/////////////////HISTOGRAM_EQUALIZATION//////////////////

equalizeHist( src, dst ); //src is gray mat

///////////DRAW_HISTOGRAM_GRAY//////////////////////
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
 
using namespace cv;
using namespace std;
 
int main()
{
    Mat imageSrc = imread("stock1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imageGrayscale;
	int width = 400, height = 400;
	int sizeHistogram = 255;
	float range[] = { 0, 255 };
	const float* histogramRange = { range };
 
	calcHist(&imageSrc, 1, 0, Mat(), imageGrayscale, 1, &sizeHistogram, &histogramRange, true, false);
 
	int bin = cvRound((double)width / sizeHistogram);
 
	Mat dispHistogram(width, height, CV_8UC3, Scalar(255, 255, 255));
 
	normalize(imageGrayscale, imageGrayscale, 0, dispHistogram.rows, NORM_MINMAX, -1, Mat());
 
	for (int i = 0; i < 255; i++) {
		line(dispHistogram, Point(bin*(i), height), Point(bin*(i), height - cvRound(imageGrayscale.at<float>(i))), Scalar(0, 0, 0), 2, 8, 0);
	}
 
    imshow("Car Stdio.vn", imageSrc);
	imshow("Hitogram", dispHistogram);
 
	// Wait input and exit
	waitKey(0);
 
	return 0;
}

////////////////DRAW_HISTOGRAM_RGB/////////////////////
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
using namespace cv;
using namespace std;
int main(int argc, char const *argv[])
{
    Mat src, dst;
	/// Load image
	src = imread( argv[1], 1 );
	vector<Mat> bgr_planes;
split( src, bgr_planes );
/// Establish the number of bins
int histSize = 256;
/// Set the ranges ( for B,G,R) )
float range[] = { 0, 256 } ;
const float * histRange = { range };
bool uniform = true; bool accumulate = false;
Mat b_hist, g_hist, r_hist;
/// Compute the histograms:
calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
// Draw the histograms for B, G and R
int hist_w = 512; int hist_h = 400;
int bin_w = cvRound( (double) hist_w/histSize );
Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
/// Normalize the
normalize(b_hist,b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
normalize(g_hist,g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
normalize(r_hist,r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );



/// Draw for each channel
for( int i = 1; i < histSize; i++ )
{
line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),Scalar( 255, 0, 0), 2, 8, 0 );
line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),Scalar( 0, 255, 0), 2, 8, 0 );
line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
Scalar( 0, 0, 255), 2, 8, 0 );
}
/// Display
namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
imshow("calcHist Demo", histImage );
waitKey(0);
return 0;
}
    

///////////////FOURIER_TRANFORM//////////////////////////

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main(int argc, char ** argv)
{
const char * filename = argc >=2 ? argv[1] : "lena.jpg";
Mat I = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
if( I.empty())
return -1;
Mat padded;
//expand input image to optimal size
int m = getOptimalDFTSize( I.rows );
int n = getOptimalDFTSize( I.cols ); // on the border add zero values
copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
Mat complexI;
merge(planes, 2, complexI);
// Add to the expanded another plane with zeros

dft(complexI, complexI);

// this way the result may fit in the source matrix

// compute the magnitude and switch to logarithmic scale
// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
split(complexI, planes);
// planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
Mat magI = planes[0];
magI += Scalar::all(1);
log(magI, magI);
// switch to logarithmic scale
// crop the spectrum, if it has an odd number of rows or columns
magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
// rearrange the quadrants of Fourier image
int cx = magI.cols/2;
int cy = magI.rows/2;

Mat q0(magI,Rect(0, 0, cx, cy));
Mat q1(magI,Rect(cx, 0, cx, cy));
Mat q2(magI,Rect(0, cy, cx, cy));
Mat q3(magI,Rect(cx, cy, cx, cy));

Mat tmp;
q0.copyTo(tmp);
q3.copyTo(q0);
tmp.copyTo(q3);
q1.copyTo(tmp);
q2.copyTo(q1);
tmp.copyTo(q2);
// swap quadrant (Top-Right with Bottom-Left)
normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
// viewable image form (float between values 0 and 1).
imshow("Input Image", I);
imshow("spectrum magnitude", magI);
waitKey();
return 0;
} 

///////////BLUR//////////////////////
    
    // i is odd interger
    
    //Normalized Block Filter - simpliest filter
        // Mat=1/(w*h)*[1,1,1,....]
    blur( src, dst, Size( i, i ), Point(-1,-1) );

    //Gaussian Filter 
        //the pixel located in the middle would have the biggest weight.
        //The weight of its neighbors decreases as the spatial distance 
        //between them and the center pixel increases.
    GaussianBlur( src, dst, Size( i, i ), 0, 0 );

    //Median Filter 
        //replace each pixel with the median of its neighboring pixels
    medianBlur ( src, dst, i );

    //Bilateral Filter
        //sometimes the filters do not only dissolve 
        //the noise, but also smooth away the edges.
        //Bilateral avoid this.
    bilateralFilter ( src, dst, i, i*2, i/2 );

///////////THRESHOLD///////////////////////
    /* 
    0: Binary
	1: Binary Inverted
	2: Threshold Truncated
	3: Threshold to Zero
	4: Threshold to Zero Inverted
	*/
threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );

//////////MAKE_BORDER//////////////////////

copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
	value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
		//if borderType is BORDER_CONSTANT
	//top, bottom, left, right is width of border (pixel)
	//borderType = BORDER_CONSTANT (0) or BORDER_REPLICATE (1)
		//use BORDER_REPLICATE to blur image without losing detail at edges

//////////SOBEL//////////////////////////

#include "opencv2/opencv.hpp"
using namespace cv;
int main( int argc, char ** argv )
{
Mat src, src_gray;
Mat grad;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
int c;
/// Load an image
src = imread( argv[1] );
if( !src.data )
{ return -1; }
GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
/// Convert it to gray
cvtColor( src, src_gray, CV_BGR2GRAY );
/// Generate grad_x and grad_y
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
/// Gradient X
//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
convertScaleAbs( grad_x, abs_grad_x );
/// Gradient Y
//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
convertScaleAbs( grad_y, abs_grad_y );
/// Total Gradient (approximate)
addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
imshow("fdasf",src_gray);
imshow( "dsaf", grad );
waitKey(0);
return 0;
}

/////////////////CANNY////////////////////

Canny( src, dst, lowThreshold, lowThreshold*ratio, kernel_size );
	//ratio recommend: 2 or 3
	//kernel_size: odd integer
	//the bigger lowThreshold, the less edges showed

/////////////////HOUGHLINE////////////////

Mat src = imread( argv[1], 1 );
Mat dst, cdst;
Canny(src, dst, 50, 200, 3);
cvtColor(dst, cdst, CV_GRAY2BGR);

	//Standard Hough Line Transform (continous line)
		vector<Vec2f> lines;
		HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );
			//(dst,lines,r_resolution(pixel),
			//		theta_resolution(rad),min_num_of_intersection,..)
		for( size_t i = 0; i < lines.size(); i++ )
		{
			float rho = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 + 1000*(-b));
			pt1.y = cvRound(y0 + 1000*(a));
			pt2.x = cvRound(x0 - 1000*(-b));
			pt2.y = cvRound(y0 - 1000*(a));
			line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
		}

	//Probabilistic Line Transform (discontinous line)
		vector<Vec4i> lines;
		HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
			//(dst,lines,r_resolution(pixel),
			//		theta_resolution(rad),min_num_of_intersection,
			//		min_num_of_point_for_a_line,max_line_gap)
		for( size_t i = 0; i < lines.size(); i++ )
		{
			Vec4i l = lines[i];
			line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
		}

imshow("source", src);
imshow("detected lines", cdst);

//////////////HOUGHCIRCLE/////////////////////////

Mat src, src_gray;
/// Read the image
src = imread( argv[1], 1 );
if( !src.data )
{ return -1; }
/// Convert it to gray
cvtColor( src, src_gray, CV_BGR2GRAY );
/// Reduce the noise so we avoid false circle detection
GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );


vector<Vec3f> circles;
/// Apply the Hough Transform to find the circles
HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );
    //HoughCircles( src_gray, circles, HOUGH_GRADIENT, 1, 
        //min_distance_between_centers, cannyThreshold, accumulatorThreshold, 0, 0 );

/// Draw the circles detected
for( size_t i = 0; i < circles.size(); i++ )
{
Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
int radius = cvRound(circles[i][2]);
// circle center
circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
// circle outline
circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
}


namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
imshow( "Hough Circle Transform Demo", src );

///////////////REMAP///////////////////////////////

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
  Mat I=imread(argv[1],1);
  Mat J=I.clone();
  Mat matx,maty;
  matx.create(I.size(),CV_32FC1);
  maty.create(I.size(),CV_32FC1);

  for( int y = 0; y < I.rows; y++ )
{ 
    for( int x = 0; x < I.cols; x++ )
    { 
        if (x<I.cols/2)
        {
          matx.at<float>(y,x)=x*2;
          maty.at<float>(y,x)=y;
        }
        else 
        {
          matx.at<float>(y,x)=0;
          maty.at<float>(y,x)=0;
        }
    }
}

remap(I,J,matx,maty,CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0));

imshow("fads",J);
  waitKey();
  return 0;
}

//////////////////FACE_AND_EYE_RECOGNIZE//////////////////

 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

 #include <iostream>
 #include <stdio.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 string window_name = "Capture - Face detection";
 RNG rng(12345);

 /** @function main */
 int main( int argc, const char** argv )
 {
    VideoCapture capture;
   Mat frame;

   //-- 1. Load the cascades
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //-- 2. Read the video stream
   if(!capture.open(0))
        return 0;

     while( true )
     {
   capture >> frame ;

   //-- 3. Apply the classifier to the frame
       if( !frame.empty() )
       { detectAndDisplay( frame ); }
       else
       { printf(" --(!) No captured frame -- Break!"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
      }
   
   return 0;
 }

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    imshow("fdasf",faceROI);  
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( size_t j = 0; j < eyes.size(); j++ )
     {
       Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
     }
  }
  //-- Show what you got
  imshow( window_name, frame );

 }
 