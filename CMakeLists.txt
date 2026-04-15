cmake_minimum_required(VERSION 3.10)
project(FlannKDTreeComparison CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Use whatever OpenCV find_package resolves.
# To point at a custom build: cmake -DOpenCV_DIR=/path/to/build ...
find_package(OpenCV 4 REQUIRED COMPONENTS core flann)

# Find the flat FLANN headers directory (where nn_index.h etc. live without prefix).
# Needed so that relative #include "nn_index.h" inside our shadow kdtree_index.h resolves.
find_path(FLANN_HEADERS_DIR
    NAMES nn_index.h
    PATHS ${OpenCV_INCLUDE_DIRS}
    PATH_SUFFIXES opencv2/flann
    NO_DEFAULT_PATH
)
if(NOT FLANN_HEADERS_DIR)
    message(FATAL_ERROR "Could not find FLANN headers dir (nn_index.h)")
endif()
message(STATUS "FLANN headers  : ${FLANN_HEADERS_DIR}")

# Common compile flags — identical for both targets (fair comparison)
set(BENCH_FLAGS -O3 -DNDEBUG)

# ── bench_original ──────────────────────────────────────────────────────────
# Uses OpenCV 4.13 headers and library as-is. No overrides.
add_executable(bench_original src/benchmark.cpp)
target_include_directories(bench_original PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bench_original PRIVATE ${OpenCV_LIBS})
target_compile_options(bench_original PRIVATE ${BENCH_FLAGS})

# ── bench_improved ──────────────────────────────────────────────────────────
# Shadows only opencv2/flann/kdtree_index.h and result_set.h with our versions.
# Every other header (nn_index.h, allocator.h, ...) comes from the 4.13 source.
add_executable(bench_improved src/benchmark.cpp)
target_include_directories(bench_improved
    BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/flann_headers   # shadows opencv2/flann/{kdtree_index,result_set}.h
        ${FLANN_HEADERS_DIR}                # lets relative "nn_index.h" etc. resolve from our shadow headers
)
target_include_directories(bench_improved PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bench_improved PRIVATE ${OpenCV_LIBS})
target_compile_options(bench_improved PRIVATE ${BENCH_FLAGS})

# ── accuracy_original ───────────────────────────────────────────────────────
add_executable(accuracy_original src/accuracy_test.cpp)
target_include_directories(accuracy_original PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(accuracy_original PRIVATE ${OpenCV_LIBS})
target_compile_options(accuracy_original PRIVATE ${BENCH_FLAGS})

# ── accuracy_improved ────────────────────────────────────────────────────────
add_executable(accuracy_improved src/accuracy_test.cpp)
target_include_directories(accuracy_improved
    BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/flann_headers
        ${FLANN_HEADERS_DIR}
)
target_include_directories(accuracy_improved PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(accuracy_improved PRIVATE ${OpenCV_LIBS})
target_compile_options(accuracy_improved PRIVATE ${BENCH_FLAGS})

# ── my_custom_tests ──────────────────────────────────────────────────────────
add_executable(my_custom_tests src/my_custom_tests.cpp)
target_include_directories(my_custom_tests
    BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/flann_headers
        ${FLANN_HEADERS_DIR}
)
target_include_directories(my_custom_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(my_custom_tests PRIVATE ${OpenCV_LIBS})
target_compile_options(my_custom_tests PRIVATE ${BENCH_FLAGS})

# ── reproduce_bug_original ───────────────────────────────────────────────────
add_executable(reproduce_bug_original src/reproduce_bug.cpp)
target_include_directories(reproduce_bug_original PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(reproduce_bug_original PRIVATE ${OpenCV_LIBS})
target_compile_options(reproduce_bug_original PRIVATE ${BENCH_FLAGS})

# ── reproduce_bug_improved ───────────────────────────────────────────────────
add_executable(reproduce_bug_improved src/reproduce_bug.cpp)
target_include_directories(reproduce_bug_improved
    BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/flann_headers
        ${FLANN_HEADERS_DIR}
)
target_include_directories(reproduce_bug_improved PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(reproduce_bug_improved PRIVATE ${OpenCV_LIBS})
target_compile_options(reproduce_bug_improved PRIVATE ${BENCH_FLAGS})

# ── bench_kmeans_original ────────────────────────────────────────────────────
add_executable(bench_kmeans_original src/benchmark_kmeans.cpp)
target_include_directories(bench_kmeans_original PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bench_kmeans_original PRIVATE ${OpenCV_LIBS})
target_compile_options(bench_kmeans_original PRIVATE ${BENCH_FLAGS})

# ── bench_kmeans_improved ─────────────────────────────────────────────────────
add_executable(bench_kmeans_improved src/benchmark_kmeans.cpp)
target_include_directories(bench_kmeans_improved
    BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/flann_headers
        ${FLANN_HEADERS_DIR}
)
target_include_directories(bench_kmeans_improved PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bench_kmeans_improved PRIVATE ${OpenCV_LIBS})
target_compile_options(bench_kmeans_improved PRIVATE ${BENCH_FLAGS})

# ── exhaustive_test_original ─────────────────────────────────────────────────
add_executable(exhaustive_test_original src/exhaustive_test.cpp)
target_include_directories(exhaustive_test_original PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(exhaustive_test_original PRIVATE ${OpenCV_LIBS})
target_compile_options(exhaustive_test_original PRIVATE ${BENCH_FLAGS})

# ── exhaustive_test_improved ──────────────────────────────────────────────────
add_executable(exhaustive_test_improved src/exhaustive_test.cpp)
target_include_directories(exhaustive_test_improved
    BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/flann_headers
        ${FLANN_HEADERS_DIR}
)
target_include_directories(exhaustive_test_improved PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(exhaustive_test_improved PRIVATE ${OpenCV_LIBS})
target_compile_options(exhaustive_test_improved PRIVATE ${BENCH_FLAGS})

message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "OpenCV include : ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "After building, run:  python3 compare.py ./build/bench_original ./build/bench_improved")
