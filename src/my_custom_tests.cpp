#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <iostream>
#include <vector>

/**
 * These functions verify that FLANN's knnSearch and radiusSearch always return
 * results sorted by distance in ascending order.
 */

bool test_knnSearch_sorting() {
    std::cout << "Running knnSearch sorting test..." << std::endl;

    const int rows = 100;
    const int cols = 8;
    const int knn = 10;

    cv::Mat data(rows, cols, CV_32F);
    cv::Mat query(1, cols, CV_32F);

    // Fill with random data
    cv::RNG rng(42);
    rng.fill(data, cv::RNG::UNIFORM, -1, 1);
    data.row(0).copyTo(query);

    // Initialize FLANN index
    cv::flann::Index index(data, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_L2);

    cv::Mat indices(1, knn, CV_32S);
    cv::Mat dists(1, knn, CV_32F);

    index.knnSearch(query, indices, dists, knn, cv::flann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED));

    for (int i = 0; i < knn; ++i) {
        float current_dist = dists.at<float>(0, i);
        if (current_dist < 0.0f) {
            std::cerr << "  Error: Distance is negative at index " << i << std::endl;
            return false;
        }
        if (i > 0) {
            float prev_dist = dists.at<float>(0, i - 1);
            if (current_dist < prev_dist) {
                std::cerr << "  Error: KNN distances are not sorted at index " << i
                          << " (prev: " << prev_dist << ", current: " << current_dist << ")" << std::endl;
                return false;
            }
        }
    }
    std::cout << "  Success: knnSearch results are sorted." << std::endl;
    return true;
}

bool test_radiusSearch_sorting() {
    std::cout << "Running radiusSearch sorting test..." << std::endl;

    cv::Mat data(100, 3, CV_32F), query(1, 3, CV_32F);
    cv::randu(data, -1.f, 1.f);
    cv::randu(query, -1.f, 1.f);

    cv::flann::Index idx(data, cv::flann::KDTreeIndexParams(1));
    cv::Mat ri, rd;

    // Using a large radius to ensure multiple points are found
    int n = idx.radiusSearch(query, ri, rd, 0.40f, 10,
                             cv::flann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED));
    std::cout<<"SR>="<<n<<" "<<ri.size()<<std::endl;
    if (n <= 1) {
        std::cerr << "  Error: Radius search failed to find enough neighbors for sorting check." << std::endl;
        return false;
    }

    for (int i = 1; i < ri.rows; ++i) {
        float current_dist = rd.at<float>(0, i);
        float prev_dist = rd.at<float>(0, i - 1);
        if (current_dist < prev_dist) {
            std::cerr << "  Error: Radius search distances are not sorted at index " << i
                      << " (prev: " << prev_dist << ", current: " << current_dist << ")" << std::endl;
            return false;
        }
    }
    std::cout << "  Success: radiusSearch results are sorted." << std::endl;
    return true;
}

int main() {
    bool knn_ok = test_knnSearch_sorting();
    bool radius_ok = test_radiusSearch_sorting();

    if (knn_ok && radius_ok) {
        std::cout << "\nAll FLANN sorting tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\nTests failed!" << std::endl;
        return -1;
    }
}