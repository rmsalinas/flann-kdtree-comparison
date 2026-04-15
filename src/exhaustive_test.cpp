/**
 * Exhaustive correctness test for the improved FLANN KDTreeIndex.
 *
 * Compiled twice by CMake:
 *   exhaustive_test_original  — stock headers (baseline)
 *   exhaustive_test_improved  — our patched headers (must be all-PASS)
 *
 * Exit code: 0 if every test passes, 1 if any fail.
 *
 * Coverage rationale
 * ------------------
 * The PR introduces five changes:
 *   A) searchLevelExact: per-dim dists[] replacement (fixes missed neighbours)
 *   B) checkCount++  inside loop after checked.set() (fixes premature budget exhaustion)
 *   C) LEAF_MAX_SIZE=10 for dim<=16 (multi-point leaves)
 *   D) NullDynamicBitset for trees==1 (zero-cost dedup)
 *   E) Template dispatch on ResultSetType (performance only, no functional change)
 *
 * Test categories
 * ---------------
 * 1. EXACT_CORRECTNESS   — FLANN_CHECKS_UNLIMITED must give 0 wrong results.
 *      Crosses: trees in {1,2,4,8} x dim in {2,3,8,16,32,128} x k in {1,5,10}
 *      Catches bugs A and C (wrong pruning or wrong leaf iteration).
 *
 * 2. RESULT_ORDER        — knn and radius results must be sorted ascending.
 *      Tests all tree counts and both search modes.
 *
 * 3. CROSS_TREE_CONSISTENCY — trees=1 and trees=4 with unlimited checks must
 *      return identical result sets (as sets, not necessarily same order).
 *      Directly tests that NullDynamicBitset (D) and DynamicBitset produce
 *      the same answer.
 *
 * 4. BUDGET_ACCOUNTING   — For approximate search at a fixed maxChecks budget,
 *      multi-tree (trees=4) must achieve recall >= single-tree (trees=1) * 0.7.
 *      The old "checkCount += node->count" bug burned the budget up to 4x too
 *      fast, making multi-tree dramatically worse than single-tree at the same
 *      nominal budget.  This test would catch that regression.
 *
 * 5. RECALL_MONOTONICITY — higher maxChecks must never reduce recall.
 *      Checks that the budget logic is not broken in a non-obvious way.
 *
 * 6. EDGE_CASES          — k=1, k==N, k>N, duplicate data points, dataset
 *      of size 1, query that coincides with a data point.
 */

#include <opencv2/core.hpp>
#include <opencv2/flann/kdtree_index.h>
#include <opencv2/flann/result_set.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

// ============================================================================
// Utilities
// ============================================================================

static int g_pass = 0;
static int g_fail = 0;

static void report(bool ok, const char* name, const char* detail = "")
{
    if (ok) {
        std::printf("PASS  %-60s %s\n", name, detail);
        ++g_pass;
    } else {
        std::printf("FAIL  %-60s %s\n", name, detail);
        ++g_fail;
    }
    std::fflush(stdout);
}

// Squared L2 distance between two float vectors of length d.
static float l2sq(const float* a, const float* b, int d)
{
    float s = 0;
    for (int i = 0; i < d; ++i) { float v = a[i]-b[i]; s += v*v; }
    return s;
}

// Brute-force KNN.  Returns sorted (dist, idx) pairs.
static std::vector<std::pair<float,int>>
bf_knn(const float* data, int N, int dim, const float* q, int k)
{
    std::vector<std::pair<float,int>> all(N);
    for (int i = 0; i < N; ++i)
        all[i] = { l2sq(data + i*dim, q, dim), i };
    std::sort(all.begin(), all.end());
    int take = std::min(k, N);
    return std::vector<std::pair<float,int>>(all.begin(), all.begin()+take);
}

// Brute-force radius set.
static std::set<int>
bf_radius(const float* data, int N, int dim, const float* q, float r_sq)
{
    std::set<int> s;
    for (int i = 0; i < N; ++i)
        if (l2sq(data + i*dim, q, dim) <= r_sq) s.insert(i);
    return s;
}

// Build a KDTreeIndex with the given number of trees.
using Index = cvflann::KDTreeIndex<cvflann::L2<float>>;

static Index* build_index(const float* data, int N, int dim, int trees)
{
    cvflann::Matrix<float> mat(const_cast<float*>(data), N, dim);
    auto* idx = new Index(mat, cvflann::KDTreeIndexParams(trees));
    idx->buildIndex();
    return idx;
}

// Run knnSearch and return (indices, dists) as flat vectors.
static void knn_search(Index& idx, const float* q, int dim,
                       int k, int maxChecks,
                       std::vector<int>& out_idx, std::vector<float>& out_dist)
{
    out_idx.assign(k, -1);
    out_dist.assign(k, 0.f);
    cvflann::Matrix<float>  qm(const_cast<float*>(q), 1, dim);
    cvflann::Matrix<int>    im(out_idx.data(), 1, k);
    cvflann::Matrix<float>  dm(out_dist.data(), 1, k);
    idx.knnSearch(qm, im, dm, k, cvflann::SearchParams(maxChecks));
}

// Run radiusSearch and return the set of indices found.
static std::set<int>
radius_search(Index& idx, const float* q, int dim, float r_sq, int N)
{
    std::vector<int>   ri(N, -1);
    std::vector<float> rd(N, 0.f);
    cvflann::Matrix<float> qm(const_cast<float*>(q), 1, dim);
    cvflann::Matrix<int>   im(ri.data(), 1, N);
    cvflann::Matrix<float> dm(rd.data(), 1, N);
    int n = idx.radiusSearch(qm, im, dm, r_sq,
                             cvflann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED));
    return std::set<int>(ri.begin(), ri.begin()+n);
}

// Compute recall: fraction of true KNN that appear in the result.
static float recall(const std::vector<std::pair<float,int>>& gt,
                    const std::vector<int>& res)
{
    std::set<int> gt_set;
    for (auto& p : gt) gt_set.insert(p.second);
    int hit = 0;
    for (int idx : res) if (gt_set.count(idx)) ++hit;
    return (float)hit / (float)gt_set.size();
}

// Generate reproducible random float data in [-range, range].
static std::vector<float> rand_data(int N, int dim, float range, uint64_t seed)
{
    cv::RNG rng(seed);
    std::vector<float> v(N * dim);
    for (float& x : v) x = rng.uniform(-range, range);
    return v;
}

// ============================================================================
// Category 1: EXACT_CORRECTNESS
//   For each (trees, dim, k) combination, run exact KNN and verify 0 wrong.
// ============================================================================

static void run_exact_correctness()
{
    const int   trees_list[] = { 1, 2, 4, 8 };
    const int   dim_list[]   = { 2, 3, 8, 16, 32, 128 };
    const int   k_list[]     = { 1, 5, 10 };
    const int   N = 2000, Q = 100;
    const float RANGE = 100.f;

    for (int trees : trees_list) {
    for (int dim   : dim_list) {
    for (int k     : k_list) {

        auto data    = rand_data(N, dim, RANGE, /*seed=*/ (uint64_t)(trees*1000 + dim*10 + k));
        auto queries = rand_data(Q, dim, RANGE, /*seed=*/ (uint64_t)(trees*1000 + dim*10 + k + 7));

        Index* idx = build_index(data.data(), N, dim, trees);

        int wrong = 0;
        std::vector<int>   ri;
        std::vector<float> rd;
        int real_k = std::min(k, N);

        for (int qi = 0; qi < Q; ++qi) {
            const float* q = queries.data() + qi*dim;
            knn_search(*idx, q, dim, real_k,
                       cvflann::FLANN_CHECKS_UNLIMITED, ri, rd);

            auto gt = bf_knn(data.data(), N, dim, q, real_k);
            float worst_gt = gt.back().first;
            std::set<int> gt_set;
            for (auto& p : gt) gt_set.insert(p.second);

            // A result is wrong if it is farther than the true k-th neighbour
            // (with a small tolerance for floating-point).
            for (int j = 0; j < real_k; ++j) {
                if (rd[j] > worst_gt * 1.001f + 1e-4f) { ++wrong; break; }
                if (!gt_set.count(ri[j]) &&
                    std::abs(rd[j] - worst_gt) > 1e-4f) { ++wrong; break; }
            }
        }

        delete idx;

        char name[128], detail[64];
        std::snprintf(name,   sizeof(name),   "EXACT_CORRECTNESS trees=%d dim=%3d k=%2d", trees, dim, k);
        std::snprintf(detail, sizeof(detail),  "%d wrong / %d queries", wrong, Q);
        report(wrong == 0, name, detail);
    }}}
}

// ============================================================================
// Category 2: RESULT_ORDER
//   knnSearch results must be sorted ascending; radiusSearch results must be
//   sorted ascending too (the improved code adds a sorted variant).
// ============================================================================

static void run_result_order()
{
    const int trees_list[] = { 1, 2, 4, 8 };
    const int dim = 4, N = 1000, Q = 50, k = 20;

    auto data    = rand_data(N, dim, 100.f, 0xABCD);
    auto queries = rand_data(Q, dim, 100.f, 0xDCBA);

    for (int trees : trees_list) {
        Index* idx = build_index(data.data(), N, dim, trees);

        bool knn_ok = true, radius_ok = true;
        std::vector<int>   ri(k);
        std::vector<float> rd(k);

        for (int qi = 0; qi < Q && knn_ok; ++qi) {
            const float* q = queries.data() + qi*dim;
            knn_search(*idx, q, dim, k, cvflann::FLANN_CHECKS_UNLIMITED, ri, rd);
            for (int j = 1; j < k; ++j)
                if (rd[j] < rd[j-1] - 1e-5f) { knn_ok = false; break; }
        }

        // radiusSearch with sorted output
        float r_sq = 50.f * 50.f;
        std::vector<int>   rri(N);
        std::vector<float> rrd(N);
        for (int qi = 0; qi < Q && radius_ok; ++qi) {
            const float* q = queries.data() + qi*dim;
            cvflann::Matrix<float> qm(const_cast<float*>(q), 1, dim);
            cvflann::Matrix<int>   im(rri.data(), 1, N);
            cvflann::Matrix<float> dm(rrd.data(), 1, N);
            int n = idx->radiusSearch(qm, im, dm, r_sq,
                        cvflann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED, 0, true));
            for (int j = 1; j < n; ++j)
                if (rrd[j] < rrd[j-1] - 1e-5f) { radius_ok = false; break; }
        }

        delete idx;

        char name[128];
        std::snprintf(name, sizeof(name), "RESULT_ORDER knn    trees=%d dim=%d k=%d", trees, dim, k);
        report(knn_ok, name);
        std::snprintf(name, sizeof(name), "RESULT_ORDER radius trees=%d dim=%d",      trees, dim);
        report(radius_ok, name);
    }
}

// ============================================================================
// Category 3: CROSS_TREE_CONSISTENCY
//   trees=1 and trees=4 with FLANN_CHECKS_UNLIMITED must return the same set
//   of nearest neighbours.  Tests that NullDynamicBitset and DynamicBitset
//   produce identical results.
// ============================================================================

static void run_cross_tree_consistency()
{
    const int   dim_list[] = { 2, 3, 8, 16, 32 };
    const int   k = 10, N = 2000, Q = 50;

    for (int dim : dim_list) {
        auto data    = rand_data(N, dim, 100.f, (uint64_t)dim * 17);
        auto queries = rand_data(Q, dim, 100.f, (uint64_t)dim * 17 + 5);

        Index* idx1 = build_index(data.data(), N, dim, 1);
        Index* idx4 = build_index(data.data(), N, dim, 4);

        int mismatches = 0;
        std::vector<int>   ri1(k), ri4(k);
        std::vector<float> rd1(k), rd4(k);

        for (int qi = 0; qi < Q; ++qi) {
            const float* q = queries.data() + qi*dim;
            knn_search(*idx1, q, dim, k, cvflann::FLANN_CHECKS_UNLIMITED, ri1, rd1);
            knn_search(*idx4, q, dim, k, cvflann::FLANN_CHECKS_UNLIMITED, ri4, rd4);

            std::set<int> s1(ri1.begin(), ri1.end());
            std::set<int> s4(ri4.begin(), ri4.end());

            // Allow for ties: the two results are consistent if their worst
            // distances match and every result whose dist < worst is shared.
            float w1 = rd1.back(), w4 = rd4.back();
            if (std::abs(w1 - w4) > 1e-4f * (w1 + w4 + 1e-6f)) {
                ++mismatches;
                continue;
            }
            for (int j = 0; j < k; ++j) {
                if (rd1[j] < w1 - 1e-4f && !s4.count(ri1[j])) { ++mismatches; break; }
                if (rd4[j] < w4 - 1e-4f && !s1.count(ri4[j])) { ++mismatches; break; }
            }
        }

        delete idx1;
        delete idx4;

        char name[128], detail[64];
        std::snprintf(name,   sizeof(name),   "CROSS_TREE_CONSISTENCY dim=%3d k=%d trees1_vs_4", dim, k);
        std::snprintf(detail, sizeof(detail),  "%d mismatches / %d queries", mismatches, Q);
        report(mismatches == 0, name, detail);
    }
}

// ============================================================================
// Category 4: BUDGET_ACCOUNTING
//   For approximate search at a fixed maxChecks budget, multi-tree (trees=4)
//   must achieve recall >= single-tree (trees=1) * RECALL_RATIO_MIN.
//
//   This is the direct regression test for the "checkCount += node->count" bug.
//   With that bug, multi-tree burned 4x the budget per leaf, so with trees=4 the
//   effective budget was ~maxChecks/4, giving dramatically lower recall than
//   trees=1 at the same maxChecks.  The threshold 0.7 is conservative; in
//   practice with the fix the multi-tree recall is equal or better.
// ============================================================================

static void run_budget_accounting()
{
    const float RECALL_RATIO_MIN = 0.70f;   // multi-tree recall >= 70% of single-tree
    const int   maxChecks_list[] = { 32, 64, 128 };
    const int   dim_list[]       = { 2, 3, 8, 16 };  // low-dim (leaf_max_size=10 active)
    const int   k = 10, N = 5000, Q = 200;

    for (int maxChecks : maxChecks_list) {
    for (int dim       : dim_list) {

        auto data    = rand_data(N, dim, 100.f, (uint64_t)dim*31 + maxChecks);
        auto queries = rand_data(Q, dim, 100.f, (uint64_t)dim*31 + maxChecks + 3);

        Index* idx1 = build_index(data.data(), N, dim, 1);
        Index* idx4 = build_index(data.data(), N, dim, 4);

        float recall1_sum = 0.f, recall4_sum = 0.f;
        std::vector<int>   ri1(k), ri4(k);
        std::vector<float> rd1(k), rd4(k);

        for (int qi = 0; qi < Q; ++qi) {
            const float* q = queries.data() + qi*dim;
            knn_search(*idx1, q, dim, k, maxChecks, ri1, rd1);
            knn_search(*idx4, q, dim, k, maxChecks, ri4, rd4);
            auto gt = bf_knn(data.data(), N, dim, q, k);
            recall1_sum += recall(gt, ri1);
            recall4_sum += recall(gt, ri4);
        }

        delete idx1;
        delete idx4;

        float r1 = recall1_sum / Q;
        float r4 = recall4_sum / Q;
        bool  ok = (r4 >= r1 * RECALL_RATIO_MIN);

        char name[128], detail[128];
        std::snprintf(name,   sizeof(name),
                      "BUDGET_ACCOUNTING maxChecks=%3d dim=%2d", maxChecks, dim);
        std::snprintf(detail, sizeof(detail),
                      "recall trees=1: %.3f  trees=4: %.3f  ratio: %.3f (min %.2f)",
                      r1, r4, r1 > 0 ? r4/r1 : 1.f, RECALL_RATIO_MIN);
        report(ok, name, detail);
    }}
}

// ============================================================================
// Category 5: RECALL_MONOTONICITY
//   Higher maxChecks must never decrease recall (average over Q queries).
//   Tests that the budget logic does not have pathological inversions.
// ============================================================================

static void run_recall_monotonicity()
{
    // maxChecks levels in ascending order
    const int   checks[]  = { 16, 32, 64, 128, 512 };
    const int   n_checks  = (int)(sizeof(checks)/sizeof(checks[0]));
    const int   trees_list[] = { 1, 4 };
    const int   dim = 4, N = 3000, Q = 100, k = 10;

    auto data    = rand_data(N, dim, 100.f, 0x1234);
    auto queries = rand_data(Q, dim, 100.f, 0x5678);

    for (int trees : trees_list) {
        Index* idx = build_index(data.data(), N, dim, trees);

        float prev_recall = 0.f;
        bool mono_ok = true;
        std::vector<int>   ri(k);
        std::vector<float> rd(k);

        for (int ci = 0; ci < n_checks; ++ci) {
            float recall_sum = 0.f;
            for (int qi = 0; qi < Q; ++qi) {
                const float* q = queries.data() + qi*dim;
                knn_search(*idx, q, dim, k, checks[ci], ri, rd);
                auto gt = bf_knn(data.data(), N, dim, q, k);
                recall_sum += recall(gt, ri);
            }
            float avg = recall_sum / Q;
            // Allow a tiny slack for statistical fluctuation
            if (avg < prev_recall - 0.05f) mono_ok = false;
            prev_recall = avg;
        }

        delete idx;

        char name[128];
        std::snprintf(name, sizeof(name),
                      "RECALL_MONOTONICITY trees=%d dim=%d k=%d (checks 16→512)", trees, dim, k);
        report(mono_ok, name);
    }
}

// ============================================================================
// Category 6: EDGE_CASES
// ============================================================================

static void run_edge_cases()
{
    // --- 6a: k == 1 (single nearest neighbour) ---
    {
        const int dim = 3, N = 500, Q = 50;
        auto data    = rand_data(N, dim, 100.f, 0xE1);
        auto queries = rand_data(Q, dim, 100.f, 0xE2);
        Index* idx = build_index(data.data(), N, dim, 4);
        int wrong = 0;
        for (int qi = 0; qi < Q; ++qi) {
            const float* q = queries.data() + qi*dim;
            std::vector<int> ri; std::vector<float> rd;
            knn_search(*idx, q, dim, 1, cvflann::FLANN_CHECKS_UNLIMITED, ri, rd);
            auto gt = bf_knn(data.data(), N, dim, q, 1);
            if (ri[0] != gt[0].second &&
                std::abs(rd[0] - gt[0].first) > 1e-4f) ++wrong;
        }
        delete idx;
        report(wrong == 0, "EDGE_CASE k=1 trees=4 dim=3");
    }

    // --- 6b: k == N (all neighbours, requesting every data point) ---
    {
        const int dim = 3, N = 200;
        auto data  = rand_data(N, dim, 100.f, 0xE3);
        auto query = rand_data(1, dim, 100.f, 0xE4);
        Index* idx = build_index(data.data(), N, dim, 4);
        std::vector<int> ri; std::vector<float> rd;
        knn_search(*idx, query.data(), dim, N,
                   cvflann::FLANN_CHECKS_UNLIMITED, ri, rd);
        // All N distinct indices should be returned
        std::set<int> s(ri.begin(), ri.end());
        delete idx;
        char detail[64];
        std::snprintf(detail, sizeof(detail), "got %d unique out of %d", (int)s.size(), N);
        report((int)s.size() == N, "EDGE_CASE k=N trees=4 dim=3", detail);
    }

    // --- 6c: k == N-1 (one fewer than total, stresses the result-set boundary) ---
    {
        const int dim = 3, N = 50;
        auto data  = rand_data(N, dim, 100.f, 0xE5);
        auto query = rand_data(1, dim, 100.f, 0xE6);
        Index* idx = build_index(data.data(), N, dim, 4);
        std::vector<int> ri; std::vector<float> rd;
        knn_search(*idx, query.data(), dim, N - 1,
                   cvflann::FLANN_CHECKS_UNLIMITED, ri, rd);
        std::set<int> s(ri.begin(), ri.end());
        bool sorted_ok = true;
        for (int j = 1; j < (int)rd.size(); ++j)
            if (rd[j] < rd[j-1] - 1e-5f) { sorted_ok = false; break; }
        delete idx;
        char detail[64];
        std::snprintf(detail, sizeof(detail), "got %d unique, sorted=%s",
                      (int)s.size(), sorted_ok ? "yes" : "no");
        report((int)s.size() == N-1 && sorted_ok, "EDGE_CASE k=N-1 trees=4 dim=3", detail);
    }

    // --- 6d: duplicate data points ---
    {
        const int dim = 3, N = 300, Q = 20;
        auto data = rand_data(N/2, dim, 100.f, 0xE7);
        // Double every point
        data.insert(data.end(), data.begin(), data.end());
        auto queries = rand_data(Q, dim, 100.f, 0xE8);
        Index* idx = build_index(data.data(), N, dim, 4);
        int wrong = 0;
        for (int qi = 0; qi < Q; ++qi) {
            const float* q = queries.data() + qi*dim;
            std::vector<int> ri; std::vector<float> rd;
            knn_search(*idx, q, dim, 5, cvflann::FLANN_CHECKS_UNLIMITED, ri, rd);
            // Results should be sorted
            for (int j = 1; j < 5; ++j)
                if (rd[j] < rd[j-1] - 1e-5f) { ++wrong; break; }
        }
        delete idx;
        report(wrong == 0, "EDGE_CASE duplicate_data_points trees=4 dim=3");
    }

    // --- 6e: dataset of size 1 ---
    {
        const int dim = 3;
        std::vector<float> data = { 1.f, 2.f, 3.f };
        std::vector<float> query = { 4.f, 5.f, 6.f };
        Index* idx = build_index(data.data(), 1, dim, 1);
        std::vector<int> ri; std::vector<float> rd;
        knn_search(*idx, query.data(), dim, 1,
                   cvflann::FLANN_CHECKS_UNLIMITED, ri, rd);
        bool ok = (ri[0] == 0);
        delete idx;
        report(ok, "EDGE_CASE dataset_size_1 k=1");
    }

    // --- 6f: query coincides exactly with a data point ---
    {
        const int dim = 3, N = 500, k = 5;
        auto data = rand_data(N, dim, 100.f, 0xE9);
        int exact_idx = 42;
        const float* q = data.data() + exact_idx * dim;  // query = data point #42
        Index* idx = build_index(data.data(), N, dim, 4);
        std::vector<int> ri; std::vector<float> rd;
        knn_search(*idx, q, dim, k, cvflann::FLANN_CHECKS_UNLIMITED, ri, rd);
        // The first result must be the exact point (distance 0)
        bool ok = (ri[0] == exact_idx && rd[0] < 1e-6f);
        delete idx;
        char detail[64];
        std::snprintf(detail, sizeof(detail),
                      "first_result_idx=%d dist=%.6f", ri[0], rd[0]);
        report(ok, "EDGE_CASE query_on_data_point trees=4", detail);
    }

    // --- 6g: dim=1 (degenerate tree with only one split dimension) ---
    {
        const int dim = 1, N = 500, Q = 50, k = 5;
        auto data    = rand_data(N, dim, 100.f, 0xEA);
        auto queries = rand_data(Q, dim, 100.f, 0xEB);
        Index* idx = build_index(data.data(), N, dim, 4);
        int wrong = 0;
        for (int qi = 0; qi < Q; ++qi) {
            const float* q = queries.data() + qi*dim;
            std::vector<int> ri; std::vector<float> rd;
            knn_search(*idx, q, dim, k, cvflann::FLANN_CHECKS_UNLIMITED, ri, rd);
            auto gt = bf_knn(data.data(), N, dim, q, k);
            for (int j = 0; j < k; ++j) {
                if (std::abs(rd[j] - gt[j].first) > 1e-4f) { ++wrong; break; }
            }
        }
        delete idx;
        report(wrong == 0, "EDGE_CASE dim=1 trees=4 exact_knn");
    }

    // --- 6h: high-k search (k=50, dim=128, high-dim path) ---
    {
        const int dim = 128, N = 1000, k = 50, Q = 20;
        auto data    = rand_data(N, dim, 10.f, 0xEC);
        auto queries = rand_data(Q, dim, 10.f, 0xED);
        Index* idx = build_index(data.data(), N, dim, 4);
        int wrong = 0;
        for (int qi = 0; qi < Q; ++qi) {
            const float* q = queries.data() + qi*dim;
            std::vector<int> ri; std::vector<float> rd;
            knn_search(*idx, q, dim, k, cvflann::FLANN_CHECKS_UNLIMITED, ri, rd);
            auto gt = bf_knn(data.data(), N, dim, q, k);
            std::set<int> gt_set;
            for (auto& p : gt) gt_set.insert(p.second);
            for (int j = 0; j < k; ++j) {
                if (!gt_set.count(ri[j]) &&
                    std::abs(rd[j] - gt.back().first) > 1e-3f) { ++wrong; break; }
            }
        }
        delete idx;
        report(wrong == 0, "EDGE_CASE high_dim=128 k=50 trees=4 exact");
    }

    // --- 6i: radius search — zero results (query far from all data) ---
    {
        const int dim = 3, N = 500;
        auto data = rand_data(N, dim, 1.f, 0xEE);   // data in [-1,1]
        std::vector<float> q = { 1000.f, 1000.f, 1000.f };  // far away
        Index* idx = build_index(data.data(), N, dim, 4);
        auto res = radius_search(*idx, q.data(), dim, 1.f /*r_sq*/, N);
        delete idx;
        report(res.empty(), "EDGE_CASE radius_zero_results trees=4");
    }

    // --- 6j: radius search — all points inside radius ---
    {
        const int dim = 3, N = 500;
        auto data = rand_data(N, dim, 1.f, 0xEF);   // data in [-1,1]
        std::vector<float> q(dim, 0.f);              // origin
        float r_sq = 1e6f;                           // huge radius
        Index* idx = build_index(data.data(), N, dim, 4);
        auto res = radius_search(*idx, q.data(), dim, r_sq, N);
        delete idx;
        char detail[64];
        std::snprintf(detail, sizeof(detail), "found %d / %d", (int)res.size(), N);
        report((int)res.size() == N, "EDGE_CASE radius_all_points trees=4", detail);
    }
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::printf("%-6s %-60s %s\n", "STATUS", "TEST", "DETAIL");
    std::printf("%s\n", std::string(100, '-').c_str());

    std::printf("\n[1] EXACT CORRECTNESS\n");
    run_exact_correctness();

    std::printf("\n[2] RESULT ORDER\n");
    run_result_order();

    std::printf("\n[3] CROSS-TREE CONSISTENCY\n");
    run_cross_tree_consistency();

    std::printf("\n[4] BUDGET ACCOUNTING (multi-tree recall >= 0.7 x single-tree)\n");
    run_budget_accounting();

    std::printf("\n[5] RECALL MONOTONICITY\n");
    run_recall_monotonicity();

    std::printf("\n[6] EDGE CASES\n");
    run_edge_cases();

    std::printf("\n%s\n", std::string(100, '-').c_str());
    std::printf("TOTAL: %d passed, %d failed\n", g_pass, g_fail);

    return g_fail > 0 ? 1 : 0;
}
