/**
 * FLANN KMeansIndex benchmark — compiled twice:
 *   bench_kmeans_original  (system OpenCV dist.h)
 *   bench_kmeans_improved  (our SIMD L2 dist.h)
 *
 * KMeans search is much more distance-dominated than KDTree: at every tree
 * level the search computes L2 to all K cluster centres.  This makes it the
 * best candidate to show whether SIMD L2 gives a real-world gain.
 *
 * Output: CSV to stdout
 *   type,branching,dim,ms
 */

#include <opencv2/core.hpp>
#include <opencv2/flann/kmeans_index.h>
#include <opencv2/flann/dist.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static const int N             = 10000;
static const int K             = 10;
static const int Q             = 500;
static const int CHECKS_APPROX = 32;
static const int NRUNS_BUILD   = 5;
static const int NRUNS_SEARCH  = 10;

// branching factors: 8 (BoW style) and 32 (default)
static const int BRANCHINGS[]  = { 8, 32 };
static const int DIMS[]        = { 3, 8, 32, 128 };

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

int main()
{
    std::printf("type,branching,dim,ms\n");
    std::fflush(stdout);

    for (int br : BRANCHINGS)
    {
        for (int dim : DIMS)
        {
            auto db_data = gen_data(N, dim, 42);
            auto q_data  = gen_data(Q, dim, 43);

            cvflann::Matrix<float> dataset(db_data.data(), N, dim);
            cvflann::Matrix<float> queries(q_data.data(),  Q, dim);

            cvflann::KMeansIndexParams params(br, /*iterations=*/11);

            // ── Build ──────────────────────────────────────────────────────
            {
                { cvflann::KMeansIndex<cvflann::L2<float>> w(dataset, params); w.buildIndex(); }

                std::vector<double> times;
                times.reserve(NRUNS_BUILD);
                for (int r = 0; r < NRUNS_BUILD; ++r) {
                    auto t0 = Clock::now();
                    cvflann::KMeansIndex<cvflann::L2<float>> idx(dataset, params);
                    idx.buildIndex();
                    times.push_back(Ms(Clock::now() - t0).count());
                }
                std::printf("build,%d,%d,%.3f\n", br, dim, median_of(times));
                std::fflush(stdout);
            }

            // Build once, reuse for search
            cvflann::KMeansIndex<cvflann::L2<float>> index(dataset, params);
            index.buildIndex();

            std::vector<int>   idx_buf(Q * K);
            std::vector<float> dst_buf(Q * K);
            cvflann::Matrix<int>   idx_mat(idx_buf.data(), Q, K);
            cvflann::Matrix<float> dst_mat(dst_buf.data(), Q, K);

            // ── KNN approximate ────────────────────────────────────────────
            {
                index.knnSearch(queries, idx_mat, dst_mat, K,
                                cvflann::SearchParams(CHECKS_APPROX));  // warm up

                std::vector<double> times;
                times.reserve(NRUNS_SEARCH);
                for (int r = 0; r < NRUNS_SEARCH; ++r) {
                    auto t0 = Clock::now();
                    index.knnSearch(queries, idx_mat, dst_mat, K,
                                    cvflann::SearchParams(CHECKS_APPROX));
                    times.push_back(Ms(Clock::now() - t0).count());
                }
                std::printf("knn_approx,%d,%d,%.3f\n", br, dim, median_of(times));
                std::fflush(stdout);
            }

            // ── KNN exact ──────────────────────────────────────────────────
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
                std::printf("knn_exact,%d,%d,%.3f\n", br, dim, median_of(times));
                std::fflush(stdout);
            }
        }
    }

    return 0;
}
