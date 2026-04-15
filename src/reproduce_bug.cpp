#include <opencv2/core.hpp>
#include <opencv2/flann/kdtree_index.h>
#include <opencv2/flann/result_set.h>
#include <iostream>
#include <vector>
#include <algorithm>

/**
 * Enhanced reproduction (2D).
 * We force many splits on X dimension by putting many points on a vertical line.
 * Then we place a query that is close to a single point but has many X-boundaries to cross.
 */

int main() {
    int N = 500;
    int dim = 2;
    std::vector<float> data(N * dim);
    
    // Most points on line X = 50, Y = [0...100]
    for (int i = 0; i < N - 1; ++i) {
        data[i*2 + 0] = 50.0f;
        data[i*2 + 1] = (float)i * 0.1f;
    }
    
    // One special point: THE TRUE NEAREST NEIGHBOR
    // Far in X (compared to the line), but query will be closer to it.
    int target_idx = N - 1;
    data[target_idx*2 + 0] = 200.0f;
    data[target_idx*2 + 1] = 0.0f;

    cvflann::Matrix<float> dataset(data.data(), N, dim);
    
    // Force many splits on X by having many points.
    // Random trees will pick high variance dimension (X or Y).
    // We use a specific seed or enough points to ensure X gets split.
    cvflann::KDTreeIndex<cvflann::L2<float>> index(dataset, cvflann::KDTreeIndexParams(1));
    index.buildIndex();

    // Query is very far to the left in X.
    // Query: X = -50, Y = 0.
    // Dist to line (X=50, Y=0) is (50 - (-50))^2 = 100^2 = 10,000
    // Dist to target (X=200, Y=0) is (200 - (-50))^2 = 250^2 = 62,500
    //
    // WAIT. If I want the bug to trigger, the query must CROSS the boundaries.
    // Let's put query BETWEEN the points.
    // Query: X = 0, Y = 0.
    // Points at X = 50. Penalty for X-split at 50 is (50-0)^2 = 2,500.
    // If we have another split at X = 100, original code adds (100-0)^2 = 10,000.
    // Sum = 12,500.
    // Correct = 10,000.
    //
    // Let's place target at X = 110. Real distance = (110-0)^2 = 12,100.
    // Since 12,500 > 12,100, the original code will prune the target!
    
    data[target_idx*2 + 0] = 110.0f;
    data[target_idx*2 + 1] = 0.0f;
    
    float qv[] = { 0.0f, 0.0f }; 
    cvflann::Matrix<float> qmat(qv, 1, dim);

    int k = 1;
    std::vector<int> indices(k);
    std::vector<float> dists(k);
    cvflann::Matrix<int> indices_mat(indices.data(), 1, k);
    cvflann::Matrix<float> dists_mat(dists.data(), 1, k);

    index.knnSearch(qmat, indices_mat, dists_mat, k, cvflann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED));

    // Brute force check
    float min_dist = 1e20f;
    int best_idx = -1;
    for(int i=0; i<N; ++i) {
        float dx = data[i*2+0] - qv[0];
        float dy = data[i*2+1] - qv[1];
        float d = dx*dx + dy*dy;
        if (d < min_dist) {
            min_dist = d;
            best_idx = i;
        }
    }

    std::cout << "Query: (" << qv[0] << ", " << qv[1] << ")" << std::endl;
    std::cout << "FLANN Result: idx=" << indices[0] << " val=(" << data[indices[0]*2] << ", " << data[indices[0]*2+1] << ") dist=" << dists[0] << std::endl;
    std::cout << "Brute Force:  idx=" << best_idx << " val=(" << data[best_idx*2] << ", " << data[best_idx*2+1] << ") dist=" << min_dist << std::endl;

    if (indices[0] != best_idx) {
        std::cout << "!!! BUG REPRODUCED: FLANN returned wrong neighbor !!!" << std::endl;
        return 1;
    } else {
        std::cout << "SUCCESS: FLANN returned correct neighbor." << std::endl;
        return 0;
    }
}
