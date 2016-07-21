#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

bool pointTrackingFlag = false;
Point2f currentPoint;

//detect mouse events
void onMouse(int event, int x, int y, int, void*)
{
	if(event == CV_EVENT_MOUSEMOVE)
	{
		cout << "(" << x << "," << y << ")" << endl;
	}
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		currentPoint = Point2f((float)x, (float)y);
		pointTrackingFlag = true;
	}
}

int main(int argc, char* argv[])
{
	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cerr << "Unable to open the webcam." << endl;
		return -1;
	}
		TermCriteria terminationCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 0.02);

		// Matching box size
		Size windowSize(25, 25);

		// Max number of points
		const int maxNumPoints = 200;

		string windowName = "Tracker";
		namedWindow(windowName, 1);
		setMouseCallback(windowName, onMouse, 0);

		Mat prevGrayImage, curGrayImage, image, frame;
		vector<Point2f> trackingPoints[2];

		// Image size scaling factor
		float scalingFactor = 1.0;

		while (true)
		{
			cap >> frame;

			if (frame.empty())
				break;

			resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

			frame.copyTo(image);

			cvtColor(image, curGrayImage, COLOR_BGR2GRAY);

			if (!trackingPoints[0].empty())
			{
				vector<uchar> statusVector;
				vector<float> errorVector;

				if (prevGrayImage.empty())
				{
					curGrayImage.copyTo(prevGrayImage);
				}

				calcOpticalFlowPyrLK(prevGrayImage, curGrayImage, trackingPoints[0], trackingPoints[1], statusVector, errorVector, windowSize, 3, terminationCriteria, 0, 0.001);

				int count = 0;
				int minDist = 7;

				for (int i = 0; i < trackingPoints[1].size(); i++)
				{
					if (pointTrackingFlag)
					{	// Check if new point are too close.
						if (norm(currentPoint - trackingPoints[1][i]) <= minDist)
						{
							pointTrackingFlag = false;
							continue;
						}
					}

					// Check if the status vector is good
					if (!statusVector[i])
						continue;

					trackingPoints[1][count++] = trackingPoints[1][i];

					// Track point icon
					int radius = 8;
					int thickness = 2;
					int lineType = 3;
					circle(image, trackingPoints[1][i], radius, Scalar(0, 255, 0), thickness, lineType);

					
				}

				trackingPoints[1].resize(count);
			}

			// Refining the location of the feature points
			if (pointTrackingFlag && trackingPoints[1].size() < maxNumPoints)
			{
				vector<Point2f> tempPoints;
				tempPoints.push_back(currentPoint);
				
				cornerSubPix(curGrayImage, tempPoints, windowSize, cvSize(-1, -1), terminationCriteria);

				trackingPoints[1].push_back(tempPoints[0]);
				pointTrackingFlag = false;
			}

			imshow(windowName, image);
			imshow("Video Input", frame);

			// ESC Check
			char ch = waitKey(10);
			if (ch == 27)
				break;

			// Update 'previous' to 'current' point vector
			std::swap(trackingPoints[1], trackingPoints[0]);

			// Update previous image to current image
			cv::swap(prevGrayImage, curGrayImage);
		}

	return 0;
}

