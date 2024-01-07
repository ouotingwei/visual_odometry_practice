#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "extra.h"

using namespace std;
using namespace cv;

// Intrinsic Parameters of a Camera
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.76, 0, 0, 1); // TUM dataset

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3 
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2 

    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    vector<DMatch> match;

    matcher->match ( descriptors_1, descriptors_2, match );

    double min_dist=10000, max_dist=0;

    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

void pose_estimation_2d_2d(vector<KeyPoint> keypoints_1,
                           vector<KeyPoint> keypoints_2,
                           vector<DMatch> matches,
                           Mat &R, Mat &t)
{

    // turn matching pairs into vector<Point2f>
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // calculate the fundamental matrix
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT);
    cout << "fundamental matrix is " << endl
         << fundamental_matrix << endl;

    // calculate the essential matrix
    Point2d principal_point(325.1, 249.7); // TUM dataset
    int focal_length = 521;                 // TUM dataset
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point, cv::RANSAC);
    cout << "Essential matrix is " << endl
         << essential_matrix << endl;

    // calculate the homography matrix
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3, cv::noArray(), 2000, 0.99);
    cout << "Homography matrix is " << endl
         << homography_matrix << endl;

    // finding rotation & translation
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point, cv::noArray());
    cout << "R is " << endl
         << R << endl;
    cout << "t is " << endl
         << t << endl;
}

int main(int argc, char **argv)
{
    Mat img_1 = imread("/home/wei/visual_odom_practice/match/test1.jpg");
    Mat img_2 = imread("/home/wei/visual_odom_practice/match/test2.jpg");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "there are " << matches.size() << " pairs of matches" << endl;

    // estimate the motion between the two img
    Mat R, t;
    pose_estimation_2d_2d(keypoints_1, keypoints_2, matches, R, t);

    // validation of E = t^R*scale
    Mat t_x = (Mat_<double>(3, 3) <<
        0, -t.at<double>(2, 0), t.at<double>(1, 0),
        t.at<double>(2, 0), 0, -t.at<double>(0, 0),
        -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    cout << "t^R = " << endl << t_x * R << endl;

    // validation of Epipolar Constraint
    for (DMatch m : matches)
    {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }

    return 0;
}
