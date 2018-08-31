#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include "iostream"
#include "stdio.h"





#include "opencv2/core.hpp"


#include "vector"
#include "stack"


#include "algorithm"
# define M_PI           3.14159265358979323846  /* pi */
# define SIZE           14  /* pi */

using namespace std;
using namespace cv;

/** Global variables */
String door_cascade_name = "myhaar.xml";
CascadeClassifier door_cascade;
string window_name = "Capture - Door detection";
RNG rng(12345);



int lbl;
void swap(double *xp, double *yp);
const int MAX = 11;
int searchNearest(int anArray[], int key);
void bubbleSort(double arr[], int n);
void search4CompNeighbors(Mat1i& image, int label, int row, int column);
int componentFinder(Mat1i& LB);
double labelArray[SIZE];
double labelArray2[SIZE];
int labelMatrix[10000][10000];
double getDistance(int a, int b, int x, int y);
void dotDetecter(int offset1, int offset2, Mat original_image);
void labeler(int offset1, int offset2, Mat gray, Mat1i imgw);

int main() {
	VideoCapture cap(1); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;
	//-- 1. Load the cascades
	if (!door_cascade.load(door_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	

	for (;;) {
		string info_string = "!";
		bool is_detected = false;
		bool is_passing = false;
		int offset = 0;
		int width = 0;
		int dist = 0;
		Mat frame;
		//-- 2. Read the video stream
		cap >> frame;
		//-- 3. Apply the classifier to the frame
		if (!frame.empty())
		{
			vector<Rect> doors;
			Mat frame_gray, dst, cdst, gray_distance;

			cvtColor(frame, frame_gray, CV_BGR2GRAY);
			cvtColor(frame, gray_distance, CV_BGR2GRAY);

			
			equalizeHist(frame_gray, frame_gray);
			

			//-- Detect doors
			door_cascade.detectMultiScale(frame_gray, doors, 16, 17, 0 | CV_HAAR_SCALE_IMAGE, Size(340,340));

			for (size_t i = 0; i < doors.size(); i++) {
				if (doors[i].x > 0) {
					
					int door_detect_left, door_detect_right, door_detect_up, door_detect_down;
					int door_left = 1080, door_right = 0, door_up, door_down;
					int door_width, door_height;

					door_detect_left = doors[i].x;
					door_detect_right = doors[i].x + doors[i].width;
					door_detect_up = doors[i].y;
					door_detect_down = doors[i].y + doors[i].height;
					door_up = door_detect_up;
					door_down = door_detect_down;

					Canny(gray_distance, dst, 50, 200, 3);
					cvtColor(dst, cdst, CV_GRAY2BGR);
					//-- Detect lines
					#if 0
					vector<Vec2f> lines;
					HoughLines(dst, lines, 1, CV_PI / 180, 100, 0, 0);

					for (size_t i = 0; i < lines.size(); i++)
					{
						float rho = lines[i][0], theta = lines[i][1];
						Point pt1, pt2;
						double a = cos(theta), b = sin(theta);
						double x0 = a*rho, y0 = b*rho;
						pt1.x = cvRound(x0 + 1000 * (-b));
						pt1.y = cvRound(y0 + 1000 * (a));
						pt2.x = cvRound(x0 - 1000 * (-b));
						pt2.y = cvRound(y0 - 1000 * (a));
						line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
					}
					#else
					vector<Vec4i> lines;
					HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 50, 10);
					for (size_t i = 0; i < lines.size(); i++)
					{
						Vec4i l = lines[i];

						if ((l[1] < door_up)) door_up = l[1];
						if ((l[3] > door_down)) door_down = l[3];
						if ((l[0] > door_detect_left) && (l[0] < door_left)) door_left = l[0];
						if ((l[2] < door_detect_right) && (l[2] > door_right)) door_right = l[2];
					}
					#endif

					door_left = (door_left + door_detect_left) / 2;
					door_right = (door_right + door_detect_right) / 2;
					
					//width !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					door_width = door_right - door_left;
					//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
					door_height = door_down - door_up;

					if (door_width > 100) {
						is_detected = true;
						try {
							Mat original_image, modified_image;

							original_image = frame.clone();
							modified_image = frame.clone();


							Rect rect(door_left, door_up, door_width, door_height);
							rectangle(frame, rect, cv::Scalar(255, 0, 0), 2, 8);

							//-- detect distance


							int array_x[10000];
							int array_y[10000];
							int x_ctr = 0;
							int y_ctr = 0;

							int sum1_x = 0;
							int sum1_y = 0;

							int sum2_x = 0;
							int sum2_y = 0;

							int distances[13] = { 1,100,120,140,200,250,300,400,505,600,19445 };


							Mat HSV;
							//Create a pointer so that we can quickly toggle between which image is being displayed
							Mat *image = &original_image;
							cvtColor(original_image, HSV, CV_BGR2HSV);
							//Check that the images loaded
							if (!original_image.data || !modified_image.data) {
								cout << "ERROR: Could not load image data." << endl;
								return -1;
							}

							//Create the display window
							//namedWindow("Unix Sample Skeleton");
							//cout << "make selection";
							//Replace center third of the image with white
							//This can be replaced with whatever filtering you need to do.
							int offset1 = image->rows;
							int offset2 = image->cols;
							//	imshow("Original Image", *image);
							/*		Mat3b bgr = original_image;

							Mat3b bgr_inv = ~bgr;
							Mat3b hsv_inv;
							cvtColor(bgr_inv, hsv_inv, COLOR_BGR2HSV);

							Mat1b mask;
							inRange(hsv_inv, Scalar(116, 0, 0), Scalar(170, 256, 256), mask); // Cyan is 90

							imshow("Mask", mask);*/
							//	modified_image=modified_image.grayScale




							dotDetecter(offset1, offset2, original_image);

							Mat gray;

							// convert RGB image to gray
							cvtColor(original_image, gray, CV_BGR2GRAY);
							Mat1i imgw = Mat1i(gray.rows, gray.cols, 0);
							labeler(offset1, offset2, gray, imgw);

							componentFinder(imgw);

							for (size_t i = 0; i < SIZE; i++)
							{
								labelArray2[i] = labelArray[i];
							}

							sort(labelArray, labelArray + SIZE);
							cout << "---------";

							int firstDot = labelArray[SIZE - 1];
							int secondDot = labelArray[SIZE - 2];

							int lbl_values[2];

							int lbl_vl_ctr = 0;
							for (size_t i = 0; i < SIZE; i++)
							{
								if (labelArray2[i] == firstDot || labelArray2[i] == secondDot) {

									lbl_values[lbl_vl_ctr] = i;
									lbl_vl_ctr++;
								}

							}

							/*
							for (int s = 0; s < lbl; s++) {
							cout << "Pixel Sizes of the pigs before Operations: "<<labelArray[s] << endl;

							}

							cout << "------------"<<endl;
							cout << "There are " << lbl << " pigs in the image" << endl;

							*/







							for (int row = 0; row < offset1; ++row) {
								for (int column = 0; column < offset2; ++column) {
									if (labelMatrix[row][column] == lbl_values[0]) {
										//cout << "a";
										sum1_x += row;
										sum1_y += column;
										x_ctr++;

									}
									if (labelMatrix[row][column] == lbl_values[1]) {

										sum2_x += row;
										sum2_y += column;
										y_ctr++;

									}

								}

							}



							try {
								double sum1x_avg = sum1_x /( x_ctr+0.1);
								double sum1y_avg = sum1_y / (x_ctr + 0.1);

								double sum2x_avg = sum2_x / (y_ctr + 0.1);
								double sum2y_avg = sum2_y / (y_ctr + 0.1);

							
							

							//cout << sum1x_avg << endl;
							//cout << sum1y_avg << endl;

							double distance2 = getDistance(sum1x_avg, sum2x_avg, sum1y_avg, sum2y_avg);

							cout << "pixel distance is= " << distance2 << endl;




							for (int a = 0; a < 10; a++) {
								for (int b = 0; b < 10; b++) {


									original_image.at<Vec3b>(sum1x_avg + a, sum1y_avg + b)[0] = 0;
									original_image.at<Vec3b>(sum1x_avg + a, sum1y_avg + b)[1] = 255;
									original_image.at<Vec3b>(sum1x_avg + a, sum1y_avg + b)[2] = 0;



									original_image.at<Vec3b>(sum2x_avg + a, sum2y_avg + b)[0] = 255;
									original_image.at<Vec3b>(sum2x_avg + a, sum2y_avg + b)[1] = 0;
									original_image.at<Vec3b>(sum2x_avg + a, sum2y_avg + b)[2] = 0;


								}


							}

							/*

							original_image.at<Vec3b>(sum1x_avg, sum1y_avg)[0] = 0;
							original_image.at<Vec3b>(sum1x_avg, sum1y_avg+1)[1] = 255;
							original_image.at<Vec3b>(sum1x_avg, sum1y_avg+2)[2] = 0;
							original_image.at<Vec3b>(sum1x_avg+1, sum1y_avg)[0] = 0;
							original_image.at<Vec3b>(sum1x_avg+1, sum1y_avg+1)[1] = 255;
							original_image.at<Vec3b>(sum1x_avg+1, sum1y_avg+2)[2] = 0;

							original_image.at<Vec3b>(sum1x_avg+2, sum1y_avg)[0] = 0;
							original_image.at<Vec3b>(sum1x_avg+2, sum1y_avg + 1)[1] = 255;
							original_image.at<Vec3b>(sum1x_avg+2, sum1y_avg + 2)[2] = 0;
							original_image.at<Vec3b>(sum1x_avg + 3, sum1y_avg)[0] = 0;
							original_image.at<Vec3b>(sum1x_avg + 3, sum1y_avg + 1)[1] = 255;
							original_image.at<Vec3b>(sum1x_avg + 3, sum1y_avg + 2)[2] = 0;

							original_image.at<Vec3b>(sum2x_avg, sum2y_avg)[0] = 51;
							original_image.at<Vec3b>(sum2x_avg, sum2y_avg + 1)[1] = 255;
							original_image.at<Vec3b>(sum2x_avg, sum2y_avg + 2)[2] = 255;
							original_image.at<Vec3b>(sum2x_avg + 1, sum2y_avg)[0] = 51;
							original_image.at<Vec3b>(sum2x_avg + 1, sum2y_avg + 1)[1] = 255;
							original_image.at<Vec3b>(sum2x_avg + 1, sum2y_avg + 2)[2] = 255;

							original_image.at<Vec3b>(sum2x_avg+2, sum2y_avg)[0] = 51;
							original_image.at<Vec3b>(sum2x_avg+2, sum2y_avg + 1)[1] = 255;
							original_image.at<Vec3b>(sum2x_avg+2, sum2y_avg + 2)[2] = 255;
							original_image.at<Vec3b>(sum2x_avg + 3, sum2y_avg)[0] = 51;
							original_image.at<Vec3b>(sum2x_avg + 3, sum2y_avg + 1)[1] = 255;
							original_image.at<Vec3b>(sum2x_avg + 3, sum2y_avg + 2)[2] = 255;*/

							//imshow("Centers", original_image);
							

							for (int i = 0; i < 3; i++) {

							}

							int nearest = searchNearest(distances, distance2);
							//cout << (nearest + 3) ;





							int can_pass[13] = { 1180,1094,1000,904, 870, 818, 754, 693, 646, 616, 605, 590, 575 };

							int act_dist = nearest + 3;

							//int nearest_laser = searchNearest(can_pass, door_width);

							int nearest_laser = 20 * door_width / can_pass[act_dist-1];

							cout << nearest_laser <<"++"<< "++" << act_dist;




							}
							catch (int e) {

								//cout << "An exception occurred. Exception Nr. " << e << '\n';
								cout << "e";

							}
							/*Display loop
							bool loop = true;
							while (loop) {
								imshow("Unix Sample Skeleton", *image);

								switch (cvWaitKey(15)) {
								case 27:  //Exit display loop if ESC is pressed
									loop = false;
									break;
								case 32:  //Swap image pointer if space is pressed
									if (image == &original_image) image = &modified_image;
									//else if (image == &modified_image) image = &hsi_image;
									else image = &original_image;
									break;
								default:
									break;
								}
							}*/

							





						}
						catch (int e) {


							cout << "E";
						}




						
					}
					else {
						// kapi yoksa yapilacak buraya @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
					}

					

				}
				else {
					// kapi yoksa yapilacak buraya @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
				}
			}

			imshow(window_name, frame);
		}
		else
		{
			printf(" --(!) No captured frame -- Break!"); break;
		}

		int c = waitKey(10);
		if ((char)c == 'c') { break; }
	}
	
}



double getDistance(int a, int b, int x, int y) {

	int calculate = (a - x)*(a - x) + (b - y)*(b - y);

	return sqrt(calculate);

}
void labeler(int offset1, int offset2, Mat gray, Mat1i imgw) {



	for (int row = 0; row < offset1; ++row) {
		for (int column = 0; column < offset2; ++column) {
			if (gray.at<uchar>(row, column) > 40) {

				imgw(row, column) = 50;

			}

		}

	}
}
void search4CompNeighbors(Mat1i& image, int label, int row, int column)
{

	labelArray[label]++;

	labelMatrix[row][column] = label;
	image(row, column) = label;



	if ((row - 1 > 0) && image(row - 1, column) == 50) {
		search4CompNeighbors(image, label, row - 1, column);
	}
	if ((row + 1 < image.rows) && image(row + 1, column) == 50) {
		search4CompNeighbors(image, label, row + 1, column);
	}
	if ((column - 1 > 0) && image(row, column - 1) == 50) {
		search4CompNeighbors(image, label, row, column - 1);
	}
	if ((column + 1 < image.cols) && image(row, column + 1) == 50) {
		search4CompNeighbors(image, label, row, column + 1);
	}



}
int componentFinder(Mat1i& image)
{
	int label = 0;
	int var = 50;

	for (int row = 0; row < image.rows; ++row) {
		for (int column = 0; column < image.cols; ++column) {
			if (image(row, column) == 50) {
				label++;
				search4CompNeighbors(image, label, row, column);

				//	cout << "label ="<<label << endl;

				var++;
			}

		}
	}
	lbl = label;
	return label;
}
void dotDetecter(int offset1, int offset2, Mat original_image) {

	double b;
	double g;
	double r;

	for (int a = 0; a < offset1; a++) {

		for (int j = 0; j < offset2; j++) {



			b = original_image.at<Vec3b>(a, j)[0];
			g = original_image.at<Vec3b>(a, j)[1];
			r = original_image.at<Vec3b>(a, j)[2];


			if (r> 190 && b <= 180 && g <= 180) {

				//	cout << original_image.at<Vec3b>(a, j)<<endl;
				//cout << "---------------------------";

				original_image.at<Vec3b>(a, j)[0] = 255;
				original_image.at<Vec3b>(a, j)[1] = 255;
				original_image.at<Vec3b>(a, j)[2] = 255;



			}
			else {
				original_image.at<Vec3b>(a, j)[0] = 0;
				original_image.at<Vec3b>(a, j)[1] = 0;
				original_image.at<Vec3b>(a, j)[2] = 0;
			}




		}

	}

}
void swap(double *xp, double *yp)
{
	double temp = *xp;
	*xp = *yp;
	*yp = temp;
}

void bubbleSort(double arr[], int n)
{
	int i, j;
	for (i = 0; i < n - 1; i++)
		for (j = 0; j < n - i - 1; j++)
			if (arr[j] > arr[j + 1])
				swap(&arr[j], &arr[j + 1]);

}




int searchNearest(int anArray[], int key)
{
	int value = abs(key - anArray[0]);
	int num = 0;

	for (int x = 0; x < MAX; x++)
	{
		if (value > abs(key - anArray[x]))
		{
			value = abs(key - anArray[x]);
			num = x;
		}
	}



	return num;

}

