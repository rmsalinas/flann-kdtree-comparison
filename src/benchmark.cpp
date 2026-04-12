/**
 * FLANN KDTree benchmark — compiled twice:
 *   bench_original  (flann_headers/original  = system OpenCV 4.5.4)
 *   bench_improved  (flann_headers/improved   = our patched headers)
 *
 * Both executables use the same random data (fixed seed) so their
 * results are directly comparable.
 *
 * Output: CSV to stdout
 *   type,dim,ms
 *   build,2,45.210
 *   knn_approx,2,12.340
 *   ...
 */

#include <opencv2/core.hpp>              // cv::RNG (data generation only), cv::AutoBuffer
#include <opencv2/flann/kdtree_index.h> // system header, or shadowed by flann_headers/
#include <opencv2/flann/result_set.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// --------------------------------------------------------------------------
// Configuration — mirrors the PR performance tests
// --------------------------------------------------------------------------

static const int  N             = 10000;    // dataset points (matches PR perf tests)
static const int  K             = 10;       // nearest neighbours
static const int  Q             = 500;      // query points (matches PR perf tests)
static const int  TREES         = 1;        // KDTree trees (same as perf tests)
static const int  CHECKS_APPROX = 32;
static const int  NRUNS_BUILD   = 5;        // repetitions (take median)
static const int  NRUNS_SEARCH  = 10;
static const int  DIMS[]        = { 2, 3, 8, 32, 128 };
static const int  NDIMS         = 5;

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

static double median_of(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static std::vector<float> gen_data(int rows, int dims, uint64_t seed)
{
    std::vector<float> buf(rows * dims);
    cv::RNG rng(seed);
    for (float& x : buf)
        x = rng.uniform(-1000.f, 1000.f);
    return buf;
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------

int main()
{
    std::printf("type,dim,ms\n");
    std::fflush(stdout);

    for (int di = 0; di < NDIMS; ++di)
    {
        const int   dim       = DIMS[di];
        const float radius    = 50.f * std::sqrt(float(dim) / 3.f);
        const float radius_sq = radius * radius;

        // Generate the same dataset for both executables
        auto db_data = gen_data(N, dim, 42);
        auto q_data  = gen_data(Q, dim, 43);

        cvflann::Matrix<float> dataset(db_data.data(), N, dim);
        cvflann::Matrix<float> queries(q_data.data(),  Q, dim);

        // ── Build ──────────────────────────────────────────────────────────
        {
            // Warm up: one untimed build to prime allocator pools and caches
            { cvflann::KDTreeIndex<cvflann::L2<float>> w(dataset, cvflann::KDTreeIndexParams(TREES)); w.buildIndex(); }

            std::vector<double> times;
            times.reserve(NRUNS_BUILD);
            for (int r = 0; r < NRUNS_BUILD; ++r) {
                auto t0 = Clock::now();
                cvflann::KDTreeIndex<cvflann::L2<float>> idx(
                    dataset, cvflann::KDTreeIndexParams(TREES));
                idx.buildIndex();
                times.push_back(Ms(Clock::now() - t0).count());
            }
            std::printf("build,%d,%.3f\n", dim, median_of(times));
            std::fflush(stdout);
        }

        // Build index once, reuse for all search benchmarks
        cvflann::KDTreeIndex<cvflann::L2<float>> index(
            dataset, cvflann::KDTreeIndexParams(TREES));
        index.buildIndex();

        std::vector<int>   idx_buf(Q * K);
        std::vector<float> dst_buf(Q * K);
        cvflann::Matrix<int>   idx_mat(idx_buf.data(), Q, K);
        cvflann::Matrix<float> dst_mat(dst_buf.data(), Q, K);

        // ── KNN approximate ────────────────────────────────────────────────
        {
            // Warm up: one untimed search to prime branch predictor and result buffers
            index.knnSearch(queries, idx_mat, dst_mat, K, cvflann::SearchParams(CHECKS_APPROX));

            std::vector<double> times;
            times.reserve(NRUNS_SEARCH);
            for (int r = 0; r < NRUNS_SEARCH; ++r) {
                auto t0 = Clock::now();
                index.knnSearch(queries, idx_mat, dst_mat, K,
                                cvflann::SearchParams(CHECKS_APPROX));
                times.push_back(Ms(Clock::now() - t0).count());
            }
            std::printf("knn_approx,%d,%.3f\n", dim, median_of(times));
            std::fflush(stdout);
        }

        // ── KNN exact ──────────────────────────────────────────────────────
        {
            index.knnSearch(queries, idx_mat, dst_mat, K,
                            cvflann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED));  // warm up

            std::vector<double> times;
            times.reserve(NRUNS_SEARCH);
            for (int r = 0; r < NRUNS_SEARCH; ++r) {
                auto t0 = Clock::now();
                index.knnSearch(queries, idx_mat, dst_mat, K,
                                cvflann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED));
                times.push_back(Ms(Clock::now() - t0).count());
            }
            std::printf("knn_exact,%d,%.3f\n", dim, median_of(times));
            std::fflush(stdout);
        }

        // ── Radius search (approximate) ────────────────────────────────────
        {
            // Pre-allocate worst-case result buffers (1 row × N cols)
            std::vector<int>   ri_buf(N);
            std::vector<float> rd_buf(N);
            cvflann::Matrix<int>   ri_mat(ri_buf.data(), 1, N);
            cvflann::Matrix<float> rd_mat(rd_buf.data(), 1, N);

            // Warm up
            for (int qi = 0; qi < Q; ++qi) {
                cvflann::Matrix<float> qrow(q_data.data() + qi * dim, 1, dim);
                index.radiusSearch(qrow, ri_mat, rd_mat, radius_sq,
                                   cvflann::SearchParams(CHECKS_APPROX));
            }

            std::vector<double> times;
            times.reserve(NRUNS_SEARCH);
            for (int r = 0; r < NRUNS_SEARCH; ++r) {
                auto t0 = Clock::now();
                for (int qi = 0; qi < Q; ++qi) {
                    cvflann::Matrix<float> qrow(q_data.data() + qi * dim, 1, dim);
                    index.radiusSearch(qrow, ri_mat, rd_mat, radius_sq,
                                       cvflann::SearchParams(CHECKS_APPROX));
                }
                times.push_back(Ms(Clock::now() - t0).count());
            }
            std::printf("radius_approx,%d,%.3f\n", dim, median_of(times));
            std::fflush(stdout);
        }
    }

    return 0;
}
