#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

int main(int argc, char **argv) {

    Mat img_1 = imread("/home/wei/visual_odom_practice/match/test1.jpg");
    Mat img_2 = imread("/home/wei/visual_odom_practice/match/test2.jpg");

    // initialize
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    // step1-detecting corner positions
    orb->detect(img_1, keypoints_1);
    orb->detect(img_2, keypoints_2);

    // step2-Calculate the BRIEF descriptor from the corner positions
    orb->compute(img_1, keypoints_1, descriptors_1);
    orb->compute(img_2, keypoints_2, descriptors_2);

    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    
    // Resize the image to half its original size
    resize(outimg1, outimg1, Size(), 0.3, 0.3);

    imshow("ORB Features @ outimg1", outimg1);

    // step3-matching the BRIEF descriptor
    std::vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);

    // step4-selecting the matching point
    double min_dist = 10000, max_dist = 0;

    // find the most likely matching point & the least likely matching point
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) {
            min_dist = dist;
        }

        if (dist > max_dist) {
            max_dist = dist;
        }
    }

    cout << "-- Max dist : " << max_dist << endl;
    cout << "-- Min dist : " << min_dist << endl;

    // Matching error: the distance between each descriptor larger than two min_dist
    // When the distance is too small, a value is needed as a lower limit
    vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    // step5-show the result
    Mat img_match, img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);

    // Resize the images to half their original size
    resize(img_match, img_match, Size(), 0.5, 0.5);
    resize(img_goodmatch, img_goodmatch, Size(), 0.5, 0.5);

    imshow("all the matching point pairs", img_match);
    imshow("Optimized matched point pairs.", img_goodmatch);

    waitKey(0);

    return 0;
}