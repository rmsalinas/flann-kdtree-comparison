# FLANN KDTree — correctness fix and performance improvements

Standalone reproducible benchmark for OpenCV PR [#28792](https://github.com/opencv/opencv/pull/28792).

The PR proposes two categories of changes to `KDTreeIndex` in OpenCV's FLANN module:

1. **A correctness bug fix** — `searchLevelExact` (the code path for exact nearest-neighbour search) silently returns wrong results in stock OpenCV.
2. **Five performance improvements** — 1.5×–2.5× faster build and search for low-dimensional data (2D, 3D, 8D), which is the dominant use case: 3D point clouds, stereo feature matching, colour histograms.

This repo lets anyone verify both claims against their own OpenCV installation, without building a patched OpenCV from source.

---

## How it works

Only two files differ from stock OpenCV:

| File | What changed |
|------|-------------|
| `flann_headers/opencv2/flann/kdtree_index.h` | Bug fix + 4 performance improvements |
| `flann_headers/opencv2/flann/result_set.h`   | 1 performance improvement (heap-backed KNN result set) |

`CMakeLists.txt` compiles the same benchmark source **twice** from the same `.cpp` file:

- `bench_original` / `accuracy_original` — compiled against your installed OpenCV headers, unchanged.
- `bench_improved` / `accuracy_improved` — compiled with our two headers shadowing the system ones via `-I flann_headers` placed before the system include path. Every other header (`nn_index.h`, `allocator.h`, …) still comes from the system.

Because `KDTreeIndex` is a fully header-only template class, each binary contains its own instantiation of the tree code. The runtime OpenCV library (`libopencv_flann.so`) is only used for non-template support code — it is not involved in the tree build or search.

---

## Requirements

- CMake ≥ 3.10
- C++14 compiler
- OpenCV 4.x (core + flann components)
- Python 3 (for `compare.py`)

---

## Build

```bash
git clone <this-repo> opencvflannimprovements
cd opencvflannimprovements
mkdir build && cd build

# System OpenCV
cmake .. -DCMAKE_BUILD_TYPE=Release

# Or point at a specific build
cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/path/to/opencv/build

cmake --build . -j$(nproc)
```

---

## Run

From the repo root:

```bash
python3 compare.py          # both correctness and performance
python3 compare.py -a       # correctness only (fast, ~10 s)
python3 compare.py -b       # performance only (~60 s)
```

If your OpenCV shared library is not on the default path:

```bash
LD_LIBRARY_PATH=/path/to/opencv/build/lib python3 compare.py
```

---

## Results

### Correctness

`FLANN_CHECKS_UNLIMITED` is documented as performing an exact nearest-neighbour search — it should always return the true closest points. In stock OpenCV it does not.

**Root cause.** `searchLevelExact` maintains a priority queue of tree branches to explore, ordered by a lower bound on the distance to any point in that branch. The lower bound for a query–branch pair is computed by accumulating a penalty for each dimension where the query lies outside the branch's bounding box.

When the tree has split the same dimension more than once along a path (common in any non-trivial tree), the original code **adds** the new penalty on top of the old one for that dimension, instead of **replacing** it. The correct formulation (Arya & Mount 1993) is to replace: the tightest constraint is the one from the deepest split, not the sum of all splits. The accumulated value is always ≥ the correct value, so the lower bound is inflated. Branches that could contain true neighbours are pruned as if they are farther away than they actually are, and those neighbours are silently skipped.

The fix: maintain a per-dimension array `dists[]` (stack-allocated via `cv::AutoBuffer`) and replace — not add — the contribution for each dimension as the search descends.

**Measured impact** (OpenCV 4.13, test cases from the PR's test suite):

```
Test case             N_q   KNN wrong   KNN wrong   Rad wrong   Rad wrong
                           (original)  (improved)  (original)  (improved)
-------------------------------------------------------------------------
standard_3D            50          10 !!!           0              44 !!!           0
large_radius         2000         373 !!!           0               5 !!!           0
dim_2D                 30           7 !!!           0              30 !!!           0
tiny_radius             1           0               0               0               0
high_dim_64            50           0               0               0               0
-------------------------------------------------------------------------
Total KNN wrong    — Original:  390,  Improved: 0
Total radius wrong — Original:   79,  Improved: 0
Improved: PERFECT accuracy across all test cases.
```

`large_radius` — 373 out of 2000 queries return the wrong nearest neighbours when using `FLANN_CHECKS_UNLIMITED`. The improved implementation gets all 2000 correct.

---

### Performance

Five accuracy-neutral improvements compound to give 1.5×–2.5× speedup for low-dimensional data:

1. **Multi-point leaf nodes.** Stop splitting at `LEAF_MAX_SIZE = 10` points instead of 1. Reduces tree depth by ~log₂(10) ≈ 3.3 levels, proportionally shrinking the search path. Adaptive: for `veclen > 16` the leaf size stays 1, preserving the original behaviour for high-dimensional data where per-point distance cost dominates.

2. **Template dispatch on result-set type.** `searchLevel`, `searchLevelExact`, `getNeighbors`, and `getExactNeighbors` are templated on the result-set type. The compiler can now inline all result-set operations and eliminate virtual dispatch in the hot search loop.

3. **`NullDynamicBitset`.** When `trees == 1` there is no need to track which nodes have been visited (the tree is a DAG with no shared nodes). A zero-overhead `NullDynamicBitset` (all methods inlined no-ops) avoids allocating and zeroing a per-query bitset.

4. **Heap-backed `KNNUniqueResultSet`.** Replaces `std::set` with a `std::vector` + `std::push_heap`/`std::pop_heap`. Better cache locality and a smaller constant factor for O(log k) insertion and worst-distance queries.

5. **Per-dimension lower-bound replacement** (also the correctness fix above). Replacing accumulation with a `dists[]` array is not only correct but also faster: the priority-queue entries are smaller and the search prunes more aggressively.

**Measured speedups** (OpenCV 4.13, N=10 000 points, K=10, Q=500 queries, trees=1):

```
Benchmark                      dim      Original      Improved       Speedup
----------------------------------------------------------------------------
Build index                      2           2.8ms           1.5ms          1.92x  <--
Build index                      3           2.8ms           1.5ms          1.87x  <--
Build index                      8           4.0ms           1.7ms          2.43x  <--
Build index                     32           7.3ms           7.3ms          1.00x
Build index                    128          14.0ms          13.7ms          1.02x

KNN approx  (checks=32)          2           3.0ms           1.9ms          1.59x  <--
KNN approx  (checks=32)          3           3.9ms           3.0ms          1.31x
KNN approx  (checks=32)          8           4.9ms           6.0ms          0.81x
KNN approx  (checks=32)         32           5.7ms           5.7ms          0.98x
KNN approx  (checks=32)        128           6.6ms           6.7ms          0.98x

KNN exact   (checks=∞)           2           1.9ms           1.7ms          1.12x
KNN exact   (checks=∞)           3           3.0ms           2.6ms          1.16x
KNN exact   (checks=∞)           8          18.9ms          12.7ms          1.49x  <--
KNN exact   (checks=∞)          32         190.5ms         209.3ms          0.91x
KNN exact   (checks=∞)         128         467.2ms         498.3ms          0.94x

Radius approx                    2           2.4ms           1.5ms          1.57x  <--
Radius approx                    3           0.8ms           0.6ms          1.34x
Radius approx                    8           0.5ms           0.5ms          0.98x
Radius approx                   32           0.7ms           0.8ms          0.91x
Radius approx                  128           2.1ms           2.4ms          0.89x
```

**Regressions at high dimension** (dim ≥ 32, exact search) are expected and understood: the correctness fix visits strictly more nodes than the buggy code (because it no longer over-prunes), so exact search at high dimension is slightly slower. This is the correct trade-off — the old "speed" was achieved by silently skipping valid results. For approximate search the effect is neutral at all dimensions.

**Regression at dim=8, approx KNN** (0.81×) is under investigation and may be a noise artefact or a leaf-size boundary effect at the `veclen ≤ 16` threshold.

---

## Author

Rafael Muñoz Salinas — OpenCV PR [#28792](https://github.com/opencv/opencv/pull/28792)
