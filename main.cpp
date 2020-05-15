#include<iostream>
#include<numeric>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>

#include "structIO.hpp"


using namespace std;
using namespace cv;

void gaussian_smoothing()
{
	Mat img;
	img = imread("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\img1gray.png");

	float gauss_data[25] = { 1, 4, 7, 4, 1,
							4, 16, 26, 16, 4,
							7, 26, 41, 26, 7,
							4, 16, 26, 16, 4,
							1, 4, 7, 4, 1 };

	Mat kernel = Mat(5, 5, CV_32F, gauss_data);

		for (int i = 0; i < 25; i++)
	{
		gauss_data[i] /= 255;
	}

	Mat result;

	filter2D(img, result, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);

	string windowname = "Gaussian Blur";
	namedWindow(windowname, 1);
	imshow(windowname, result);
	waitKey(0);
}

void magnitude_sobel()
{
	Mat img;
	img = imread("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\img1gray.png");


	Mat imgGray;
	cv::cvtColor(img, imgGray, COLOR_BGR2GRAY);

	Mat blurred = imgGray.clone();
	int filtersize = 5;
	int stddev = 2.0;
	GaussianBlur(imgGray, blurred, Size(filtersize, filtersize), stddev);

	float sobel_x[9] = { -1,0,+1,-2,0,+2,-1,0,+1 };
	Mat kernel_x = Mat(3, 3,CV_32F, sobel_x);
	float sobel_y[9] = { -1, -2, -1, 0, 0, 0, +1, +2, +1 };
	cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, sobel_y);

	Mat result_x, result_y;
	filter2D(blurred, result_x, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	filter2D(blurred, result_y, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	Mat magnitude = imgGray.clone();
	for (int r = 0; r < magnitude.rows; r++)
	{
		for (int c = 0; c < magnitude.cols; c++)
		{
			magnitude.at<unsigned char>(r, c) = sqrt(pow(result_x.at<unsigned char>(r, c), 2) +
				pow(result_y.at<unsigned char>(r, c), 2));
		}

	}

	string windowName = "Gaussian Blurring";
	cv::namedWindow(windowName, 1); // create window
	cv::imshow(windowName, magnitude);
	cv::waitKey(0); // wait for keyboard input before continuing

}

void cornernessHarris()
{
	// load image from file
	cv::Mat img;
	img = cv::imread("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\img1.png");
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

	// Detector parameters
	int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
	int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
	int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
	double k = 0.04;       // Harris parameter (see equation for details)

	// Detect Harris corners and normalize output
	cv::Mat dst, dst_norm, dst_norm_scaled;
	dst = cv::Mat::zeros(img.size(), CV_32FC1);
	cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
	cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	cv::convertScaleAbs(dst_norm, dst_norm_scaled);

	// visualize results
	string windowName = "Harris Corner Detector Response Matrix";
	cv::namedWindow(windowName, 4);
	cv::imshow(windowName, dst_norm_scaled);
	cv::waitKey(0);

	// Look for prominent corners and instantiate keypoints
	vector<cv::KeyPoint> keypoints;
	double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
	for (size_t j = 0; j < dst_norm.rows; j++)
	{
		for (size_t i = 0; i < dst_norm.cols; i++)
		{
			int response = (int)dst_norm.at<float>(j, i);
			if (response > minResponse)
			{ // only store points above a threshold

				cv::KeyPoint newKeyPoint;
				newKeyPoint.pt = cv::Point2f(i, j);
				newKeyPoint.size = 2 * apertureSize;
				newKeyPoint.response = response;

				// perform non-maximum suppression (NMS) in local neighbourhood around new key point
				bool bOverlap = false;
				for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
				{
					double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
					if (kptOverlap > maxOverlap)
					{
						bOverlap = true;
						if (newKeyPoint.response > (*it).response)
						{                      // if overlap is >t AND response is higher for new kpt
							*it = newKeyPoint; // replace old key point with new one
							break;             // quit loop over keypoints
						}
					}
				}
				if (!bOverlap)
				{                                     // only add new key point if no overlap has been found in previous NMS
					keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
				}
			}
		} // eof loop over cols
	}     // eof loop over rows

	// visualize keypoints
	windowName = "Harris Corner Detection Results";
	cv::namedWindow(windowName, 5);
	cv::Mat visImage = dst_norm_scaled.clone();
	cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow(windowName, visImage);
	cv::waitKey(0);
	
}

void detKeypoints()
{
	Mat imgGray;
	Mat img = imread("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\img1.png");
	cvtColor(img, imgGray, COLOR_BGR2GRAY);

	/// Shi-tomasi parameters

	int blocksize = 6;
	double maxOverlap = 0.0;
	double minDistance = (1.0 - maxOverlap)*blocksize;
	int maxCorners = img.rows*img.cols / max(1.0, minDistance);
	double qualityLevel = 0.01;
	double k = 0.04;
	bool useHarris = false;

	vector<KeyPoint> kptsShiTomasi;
	vector<Point2f> corners;
	double t = (double)getTickCount();
	goodFeaturesToTrack(imgGray, corners, maxCorners, qualityLevel, minDistance, Mat());
	t = (double)getTickCount() - t / getTickFrequency();
	cout << "Shi-Tomasi with n= " << corners.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

	for (auto it = corners.begin(); it != corners.end(); it++)
	{
		KeyPoint newKeyPoint;
		newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
		newKeyPoint.size = blocksize;
		kptsShiTomasi.push_back(newKeyPoint);
	}

	cv::Mat visImage = img.clone();
	cv::drawKeypoints(img, kptsShiTomasi, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	string windowName = "Shi-Tomasi Results";
	cv::namedWindow(windowName, 1);
	cv::imshow(windowName, visImage);
	waitKey(0);

	//Brisk
	cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
	vector<cv::KeyPoint> kptsBRISK;

	double time = (double)cv::getTickCount();
	detector->detect(imgGray, kptsBRISK);
	time = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "BRISK detector with n= " << kptsBRISK.size() << " keypoints in " << 1000 * time / 1.0 << " ms" << endl;

	cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
	cv::Mat descBRISK;
	t = (double)cv::getTickCount();
	descriptor->compute(imgGray, kptsBRISK, descBRISK);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "BRISK descriptor in " << 1000 * t / 1.0 << " ms" << endl;

	// visualize results
	cv::Mat visualiseImage = img.clone();
	cv::drawKeypoints(img, kptsBRISK, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	string windowName1 = "BRISK Results";
	cv::namedWindow(windowName, 2);
	imshow(windowName1, visualiseImage);
	waitKey(0);

	//detector = xfeatures2d::SIFT::create();
	//vector<cv::KeyPoint> kptsSIFT;

	//t = (double)cv::getTickCount();
	//detector->detect(imgGray, kptsSIFT);
	//t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	//cout << "SIFT detector with n= " << kptsSIFT.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

	//descriptor = cv::xfeatures2d::SiftDescriptorExtractor::create();
	//cv::Mat descSIFT;
	//t = (double)cv::getTickCount();
	//descriptor->compute(imgGray, kptsSIFT, descSIFT);
	//t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	//cout << "SIFT descriptor in " << 1000 * t / 1.0 << " ms" << endl;

	//visImage = img.clone();
	//cv::drawKeypoints(img, kptsSIFT, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//windowName = "SIFT Results";
	//cv::namedWindow(windowName, 3);
	//imshow(windowName, visImage);
	//cv::waitKey(0);

}

void descKeypoints1()
{
	// load image from file and convert to grayscale
	cv::Mat imgGray;
	cv::Mat img = cv::imread("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\img1.png");
	cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

	// BRISK detector / descriptor
	cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
	vector<cv::KeyPoint> kptsBRISK;

	double t = (double)cv::getTickCount();
	detector->detect(imgGray, kptsBRISK);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "BRISK detector with n= " << kptsBRISK.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

	cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
	cv::Mat descBRISK;
	t = (double)cv::getTickCount();
	descriptor->compute(imgGray, kptsBRISK, descBRISK);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "BRISK descriptor in " << 1000 * t / 1.0 << " ms" << endl;

	// visualize results
	cv::Mat visImage = img.clone();
	cv::drawKeypoints(img, kptsBRISK, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	string windowName = "BRISK Results";
	cv::namedWindow(windowName, 1);
	imshow(windowName, visImage);
	cv::waitKey(0);


	//detector = cv::xfeatures2d::SIFT::create(); 
	//vector<cv::KeyPoint> kptsSIFT;

	//t = (double)cv::getTickCount();
	//detector->detect(imgGray, kptsSIFT);
	//t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	//cout << "SIFT detector with n= " << kptsSIFT.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

	//descriptor = cv::xfeatures2d::SiftDescriptorExtractor::create();
	//cv::Mat descSIFT;
	//t = (double)cv::getTickCount();
	//descriptor->compute(imgGray, kptsSIFT, descSIFT);
	//t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	//cout << "SIFT descriptor in " << 1000 * t / 1.0 << " ms" << endl;

	//visImage = img.clone();
	//cv::drawKeypoints(img, kptsSIFT, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//windowName = "SIFT Results";
	//cv::namedWindow(windowName, 2);
	//imshow(windowName, visImage);
	//cv::waitKey(0);
}


void matchDescriptors(cv::Mat &imgSource, cv::Mat &imgRef, vector<cv::KeyPoint> &kPtsSource, vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
	vector<cv::DMatch> &matches, string descriptorType, string matcherType, string selectorType)
{

	// configure matcher
	bool crossCheck = false;
	cv::Ptr<cv::DescriptorMatcher> matcher;

	if (matcherType.compare("MAT_BF") == 0)
	{

		int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
		matcher = cv::BFMatcher::create(normType, crossCheck);
		cout << "BF matching cross-check=" << crossCheck;
	}
	else if (matcherType.compare("MAT_FLANN") == 0)
	{
		if (descSource.type() != CV_32F)
		{ // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
			descSource.convertTo(descSource, CV_32F);
			descRef.convertTo(descRef, CV_32F);
		}

		matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
		cout << "FLANN matching";
	}

	// perform matching task=
	if (selectorType.compare("SEL_NN") == 0)
	{ // nearest neighbor (best match)

		double t = (double)cv::getTickCount();
		matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
	}
	else if (selectorType.compare("SEL_KNN") == 0)
	{ // k nearest neighbors (k=2)

		vector<vector<cv::DMatch>> knn_matches;
		double t = (double)cv::getTickCount();
		matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

		// STUDENT TASK
		// filter matches using descriptor distance ratio test
		double minDescDistRatio = 0.8;
		for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
		{

			if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
			{
				matches.push_back((*it)[0]);
			}
		}
		cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
	
	}

	// visualize results
	cv::Mat matchImg = imgRef.clone();
	cv::drawMatches(imgSource, kPtsSource, imgRef, kPtsRef, matches,
		matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	string windowName = "Matching keypoints between two camera images (best 50)";
	cv::namedWindow(windowName, 7);
	cv::imshow(windowName, matchImg);
	cv::waitKey(0);
}




int main()
{
	//gaussian_smoothing();
	//magnitude_sobel();
	//cornernessHarris();
	//detKeypoints();
	/*Mat imgSource = imread("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\img1gray.png");
	Mat imgRef =  imread("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\img2gray.png");

	vector<cv::KeyPoint> kptsSource, kptsRef;
	readKeypoints("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\C35A5_KptsSource_BRISK_large.dat", kptsSource);
	readKeypoints("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\C35A5_KptsRef_BRISK_large.dat", kptsRef);

	Mat descSource, descRef;
	readDescriptors("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\C35A5_DescSource_BRISK_large.dat", descSource);
	readDescriptors("C:\\Users\\SandeepMaddula\\source\\repos\\opencv\\opencv\\C35A5_DescSource_BRISK_large.dat", descRef);

	vector<DMatch> matches;

	string matcherType = "MAT_BF";
	string descriptorType = "DES_BINARY";
	string selectorType = "SEL_NN";
	matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType);
*/
	descKeypoints1();
}