/**
 * FLANN KDTree accuracy test — compiled twice:
 *   accuracy_original  (flann_headers/original = stock OpenCV 4.13)
 *   accuracy_improved  (flann_headers/improved = our patched headers)
 *
 * Demonstrates the correctness bug in the original searchLevelExact:
 * lower-bound contributions are *accumulated* per dimension instead of
 * *replaced*, inflating bounds and causing valid neighbours to be pruned.
 * This means knnSearch with FLANN_CHECKS_UNLIMITED can silently return
 * wrong results.
 *
 * Test strategy: same test cases as the PR's test_kdtree.cpp.
 * For each case we run exact KNN and compare against brute-force L2.
 *
 * Output: CSV to stdout
 *   case,n_queries,wrong_exact,wrong_radius
 */

#include <opencv2/core.hpp>
#include <opencv2/flann/kdtree_index.h>
#include <opencv2/flann/result_set.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

// --------------------------------------------------------------------------
// Brute-force helpers
// --------------------------------------------------------------------------

static float l2sq(const float* a, const float* b, int d)
{
    float s = 0;
    for (int i = 0; i < d; ++i) { float v = a[i] - b[i]; s += v * v; }
    return s;
}

struct BFResult {
    std::vector<int>   knn_idx;
    std::vector<float> knn_dist;
    std::set<int>      radius_set;
};

static BFResult brute_force(const float* data, int N, int dim,
                             const float* q, int k, float radius_sq)
{
    std::vector<std::pair<float,int>> all(N);
    for (int i = 0; i < N; ++i)
        all[i] = { l2sq(data + i*dim, q, dim), i };
    std::sort(all.begin(), all.end());

    BFResult r;
    int take = std::min(k, N);
    r.knn_idx.resize(take);
    r.knn_dist.resize(take);
    for (int i = 0; i < take; ++i) {
        r.knn_idx[i]  = all[i].second;
        r.knn_dist[i] = all[i].first;
    }
    for (auto& p : all)
        if (p.first <= radius_sq) r.radius_set.insert(p.second);
    return r;
}

// --------------------------------------------------------------------------
// Counting helpers  (same logic as test_kdtree.cpp)
// --------------------------------------------------------------------------

// Returns number of query results that are worse than the true k-th neighbor
static int count_exact_wrong(const float* data, int N, int dim,
                              const float* queries, int Q, int k,
                              const int* idx_out, const float* dist_out,
                              float radius_sq)
{
    int wrong = 0;
    for (int qi = 0; qi < Q; ++qi) {
        BFResult bf = brute_force(data, N, dim, queries + qi*dim, k, radius_sq);
        float worst_bf = bf.knn_dist.back();
        std::set<int> gt(bf.knn_idx.begin(), bf.knn_idx.end());

        const int*   ri = idx_out  + qi * k;
        const float* rd = dist_out + qi * k;

        for (int j = 0; j < k; ++j) {
            float d = rd[j];
            if (d > worst_bf * 1.001f + 1e-4f) { ++wrong; break; }
            if (gt.find(ri[j]) == gt.end() &&
                std::abs(d - worst_bf) > 1e-4f) { ++wrong; break; }
        }
    }
    return wrong;
}

static int count_radius_wrong(const float* data, int N, int dim,
                               const float* queries, int Q,
                               cvflann::KDTreeIndex<cvflann::L2<float>>& index,
                               float radius_sq)
{
    int wrong = 0;
    std::vector<int>   ri_buf(N);
    std::vector<float> rd_buf(N);
    cvflann::Matrix<int>   ri_mat(ri_buf.data(), 1, N);
    cvflann::Matrix<float> rd_mat(rd_buf.data(), 1, N);

    for (int qi = 0; qi < Q; ++qi) {
        BFResult bf = brute_force(data, N, dim, queries + qi*dim, 1, radius_sq);

        cvflann::Matrix<float> qrow(
            const_cast<float*>(queries) + qi * dim, 1, dim);
        int n = index.radiusSearch(qrow, ri_mat, rd_mat, radius_sq,
                                   cvflann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED));

        std::set<int> cv_set;
        for (int j = 0; j < n; ++j) cv_set.insert(ri_buf[j]);
        if (cv_set != bf.radius_set) ++wrong;
    }
    return wrong;
}

// --------------------------------------------------------------------------
// Test cases  (same parameters as test_kdtree.cpp)
// --------------------------------------------------------------------------

struct TestCase {
    const char* name;
    int   N, dim, k, n_queries;
    float radius;
    unsigned seed;
    float dist_range;
};

static const TestCase kCases[] = {
    // Cases most affected by the lower-bound accumulation bug:
    { "standard_3D",    10000,  3,  10,   50,  200, 42, 1000 },
    { "large_radius",    5000,  3,  10, 2000,   50, 37, 1000 },
    { "dim_2D",          8000,  2,  10,   30,  200, 53,  500 },
    { "tiny_radius",     5000,  3,  10,    1,  100, 41, 1000 },
    // Higher-dim cases — bug is less visible but still present
    { "high_dim_64",     2000, 64,  10,   50,   50, 31,   10 },
};
static const int N_CASES = (int)(sizeof(kCases) / sizeof(kCases[0]));

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------

int main()
{
    std::printf("case,n_queries,wrong_exact,wrong_radius\n");
    std::fflush(stdout);

    for (int ci = 0; ci < N_CASES; ++ci) {
        const TestCase& tc = kCases[ci];

        // Generate data with cv::RNG (same seeding as test_kdtree.cpp)
        cv::RNG rng(tc.seed);
        std::vector<float> data(tc.N * tc.dim);
        for (float& x : data) x = rng.uniform(-tc.dist_range, tc.dist_range);

        // Mix: 1/4 exact-match queries (data points), 3/4 random
        cv::RNG rng2(tc.seed + 1);
        std::vector<float> queries(tc.n_queries * tc.dim);
        for (int qi = 0; qi < tc.n_queries; ++qi) {
            float* q = queries.data() + qi * tc.dim;
            if (qi < tc.n_queries / 4) {
                int src = (int)(rng2.next() % tc.N);
                const float* src_ptr = data.data() + src * tc.dim;
                std::copy(src_ptr, src_ptr + tc.dim, q);
            } else {
                for (int d = 0; d < tc.dim; ++d)
                    q[d] = rng2.uniform(-tc.dist_range, tc.dist_range);
            }
        }

        float radius_sq = tc.radius * tc.radius;
        int   real_k    = std::min(tc.k, tc.N);

        cvflann::Matrix<float> dataset(data.data(),    tc.N,         tc.dim);
        cvflann::Matrix<float> qmat   (queries.data(), tc.n_queries, tc.dim);

        cvflann::KDTreeIndex<cvflann::L2<float>> index(
            dataset, cvflann::KDTreeIndexParams(1));
        index.buildIndex();

        // Exact KNN
        std::vector<int>   idx_buf(tc.n_queries * real_k);
        std::vector<float> dst_buf(tc.n_queries * real_k);
        cvflann::Matrix<int>   idx_mat(idx_buf.data(), tc.n_queries, real_k);
        cvflann::Matrix<float> dst_mat(dst_buf.data(), tc.n_queries, real_k);

        index.knnSearch(qmat, idx_mat, dst_mat, real_k,
                        cvflann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED));

        int wrong_exact  = count_exact_wrong(data.data(),    tc.N,    tc.dim,
                                             queries.data(), tc.n_queries, real_k,
                                             idx_buf.data(), dst_buf.data(),
                                             radius_sq);
        int wrong_radius = count_radius_wrong(data.data(),    tc.N,    tc.dim,
                                              queries.data(), tc.n_queries,
                                              index, radius_sq);

        std::printf("%s,%d,%d,%d\n",
                    tc.name, tc.n_queries, wrong_exact, wrong_radius);
        std::fflush(stdout);
    }

    return 0;
}
