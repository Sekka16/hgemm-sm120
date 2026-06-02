#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "hgemm.cu"

// -------------------- 错误检查宏 --------------------
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err__));        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t stat__ = (call);                                         \
        if (stat__ != CUBLAS_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                 \
                         __FILE__, __LINE__, static_cast<int>(stat__));         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// -------------------- host 端 half 工具 --------------------
static inline half float_to_half_host(float x) {
    return __float2half(x);
}

struct CompareStats {
    double max_abs_err;
    double avg_abs_err;
    double max_rel_err;
    int idx_max;
};

struct TileConfig {
    int bm;
    int bn;
    int bk;
};

struct BenchResult {
    const char* kernel_name;
    TileConfig tile;
    float avg_ms;
    double tflops;
    CompareStats stats;
    bool valid;
};

// -------------------- 初始化 --------------------
void init_matrix_half(std::vector<half>& mat, float scale = 1.0f, uint32_t seed = 12345) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = float_to_half_host(dist(rng));
    }
}

// -------------------- 结果对比 --------------------
CompareStats compare_results(const std::vector<float>& ref,
                             const std::vector<float>& out,
                             int M, int N) {
    CompareStats stats = {0.0, 0.0, 0.0, -1};

    const int total = M * N;
    for (int i = 0; i < total; ++i) {
        const double a = static_cast<double>(ref[i]);
        const double b = static_cast<double>(out[i]);
        const double abs_err = std::abs(a - b);
        const double rel_err = abs_err / (std::abs(a) + 1e-6);

        stats.avg_abs_err += abs_err;
        if (abs_err > stats.max_abs_err) {
            stats.max_abs_err = abs_err;
            stats.idx_max = i;
        }
        if (rel_err > stats.max_rel_err) {
            stats.max_rel_err = rel_err;
        }
    }

    stats.avg_abs_err /= static_cast<double>(total);

    std::printf("max_abs_err = %.8f\n", stats.max_abs_err);
    std::printf("avg_abs_err = %.8f\n", stats.avg_abs_err);
    std::printf("max_rel_err = %.8f\n", stats.max_rel_err);

    if (stats.idx_max >= 0) {
        const int r = stats.idx_max / N;
        const int c = stats.idx_max % N;
        std::printf("worst at (%d, %d): ref = %.8f, out = %.8f\n",
                    r, c, ref[stats.idx_max], out[stats.idx_max]);
    }

    return stats;
}

// -------------------- 自定义 kernel launch --------------------
void launch_custom_hgemm(const half* dA,
                         const half* dB,
                         float* dC,
                         int M, int N, int K,
                         int tile_bm,
                         int tile_bn,
                         int tile_bk) {
    dispatch_hgemm_wmma(dA, dB, dC, M, N, K, tile_bm, tile_bn, tile_bk);

    CHECK_CUDA(cudaGetLastError());
}

// -------------------- cuBLAS --------------------
// row-major C = A * B
// 等价转成列主序：C^T = B * A
void launch_cublas_hgemm(cublasHandle_t handle,
                         const half* dA,
                         const half* dB,
                         float* dC,
                         int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        dB, CUDA_R_16F, N,
        dA, CUDA_R_16F, K,
        &beta,
        dC, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

float benchmark_custom_hgemm(const half* dA,
                             const half* dB,
                             float* dC,
                             int M, int N, int K,
                             int tile_bm,
                             int tile_bn,
                             int tile_bk,
                             int warmup_iters,
                             int repeat_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        launch_custom_hgemm(dA, dB, dC, M, N, K, tile_bm, tile_bn, tile_bk);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat_iters; ++i) {
        launch_custom_hgemm(dA, dB, dC, M, N, K, tile_bm, tile_bn, tile_bk);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_ms / static_cast<float>(repeat_iters);
}

float benchmark_cublas_hgemm(cublasHandle_t handle,
                             const half* dA,
                             const half* dB,
                             float* dC,
                             int M, int N, int K,
                             int warmup_iters,
                             int repeat_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        launch_cublas_hgemm(handle, dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat_iters; ++i) {
        launch_cublas_hgemm(handle, dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_ms / static_cast<float>(repeat_iters);
}

int main() {
    const int M = 8192;
    const int N = 8192;
    const int K = 8192;

    const int warmup_iters = 2;
    const int repeat_iters = 10;

    std::printf("Test shape: M=%d, N=%d, K=%d\n", M, N, K);
    std::printf("Benchmark: warmup=%d, repeat=%d\n", warmup_iters, repeat_iters);

    if (K % 16 != 0) {
        std::fprintf(stderr,
                     "Shape not aligned with current kernel assumptions. "
                     "Require K %% 16 == 0.\n");
        return EXIT_FAILURE;
    }

    std::vector<half> hA(static_cast<size_t>(M) * K);
    std::vector<half> hB(static_cast<size_t>(K) * N);
    std::vector<float> hC_custom(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> hC_cublas(static_cast<size_t>(M) * N, 0.0f);

    init_matrix_half(hA, 1.0f, 12345);
    init_matrix_half(hB, 1.0f, 67890);

    half* dA = nullptr;
    half* dB = nullptr;
    float* dC_custom = nullptr;
    float* dC_cublas = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeof(half) * static_cast<size_t>(M) * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(half) * static_cast<size_t>(K) * N));
    CHECK_CUDA(cudaMalloc(&dC_custom, sizeof(float) * static_cast<size_t>(M) * N));
    CHECK_CUDA(cudaMalloc(&dC_cublas, sizeof(float) * static_cast<size_t>(M) * N));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(),
                          sizeof(half) * static_cast<size_t>(M) * K,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(),
                          sizeof(half) * static_cast<size_t>(K) * N,
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // -------------------- 正式运行 --------------------
    CHECK_CUDA(cudaMemset(dC_cublas, 0, sizeof(float) * static_cast<size_t>(M) * N));
    launch_cublas_hgemm(handle, dA, dB, dC_cublas, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC_cublas.data(), dC_cublas,
                          sizeof(float) * static_cast<size_t>(M) * N,
                          cudaMemcpyDeviceToHost));

    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const float cublas_avg_ms = benchmark_cublas_hgemm(
        handle,
        dA, dB, dC_cublas,
        M, N, K,
        warmup_iters,
        repeat_iters
    );
    const double cublas_tflops = flops / (static_cast<double>(cublas_avg_ms) * 1.0e9);

    const TileConfig tile_configs[] = {
        {64, 64, 16},
        {64, 64, 32},
        {64, 64, 64},
        {64, 128, 16},
        {64, 128, 32},
        {64, 256, 16},
        {64, 256, 32},
        {128, 64, 16},
        {128, 64, 32},
        {128, 128, 16},
        {128, 128, 32},
        {128, 256, 16},
        {128, 256, 32},
        {256, 64, 16},
        {256, 64, 32},
        {256, 128, 16},
        {256, 128, 32},
    };
    std::vector<BenchResult> results;
    BenchResult best = {
        "cublas",
        {0, 0, 0},
        cublas_avg_ms,
        cublas_tflops,
        {0.0, 0.0, 0.0, -1},
        true
    };
    results.push_back(best);

    BenchResult best_custom = {
        "",
        {0, 0, 0},
        std::numeric_limits<float>::infinity(),
        0.0,
        {0.0, 0.0, 0.0, -1},
        false
    };

    for (const TileConfig tile : tile_configs) {
        if (M % tile.bm != 0 || N % tile.bn != 0 || K % tile.bk != 0) {
            std::printf("\nSkip wmma BM=%d BN=%d BK=%d: shape not aligned.\n",
                        tile.bm, tile.bn, tile.bk);
            continue;
        }

        std::printf("\nRun wmma BM=%d BN=%d BK=%d:\n",
                    tile.bm, tile.bn, tile.bk);

        CHECK_CUDA(cudaMemset(dC_custom, 0, sizeof(float) * static_cast<size_t>(M) * N));
        launch_custom_hgemm(
            dA, dB, dC_custom,
            M, N, K,
            tile.bm, tile.bn, tile.bk
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(hC_custom.data(), dC_custom,
                              sizeof(float) * static_cast<size_t>(M) * N,
                              cudaMemcpyDeviceToHost));

        CompareStats stats = compare_results(hC_cublas, hC_custom, M, N);
        const bool valid = stats.max_abs_err <= 0.5 && stats.avg_abs_err <= 0.02;
        std::printf("valid = %s\n", valid ? "yes" : "no");

        const float avg_ms = benchmark_custom_hgemm(
            dA, dB, dC_custom,
            M, N, K,
            tile.bm, tile.bn, tile.bk,
            warmup_iters,
            repeat_iters
        );
        const double tflops = flops / (static_cast<double>(avg_ms) * 1.0e9);

        std::printf("avg_ms = %.4f, TFLOPS = %.2f\n", avg_ms, tflops);

        BenchResult result = {"wmma", tile, avg_ms, tflops, stats, valid};
        results.push_back(result);

        if (valid && avg_ms < best.avg_ms) {
            best = result;
        }
        if (valid && avg_ms < best_custom.avg_ms) {
            best_custom = result;
        }
    }

    std::sort(results.begin(), results.end(),
              [](const BenchResult& a, const BenchResult& b) {
                  return a.tflops > b.tflops;
              });

    std::printf("\nSummary:\n");
    std::printf("%-6s %-6s %-6s %-6s %-10s %-10s %-10s %-8s\n",
                "kernel", "BM", "BN", "BK", "avg_ms", "TFLOPS", "max_abs", "valid");
    for (const BenchResult& result : results) {
        char bm_buf[16];
        char bn_buf[16];
        char bk_buf[16];
        if (std::strcmp(result.kernel_name, "cublas") == 0) {
            std::snprintf(bm_buf, sizeof(bm_buf), "-");
            std::snprintf(bn_buf, sizeof(bn_buf), "-");
            std::snprintf(bk_buf, sizeof(bk_buf), "-");
        } else {
            std::snprintf(bm_buf, sizeof(bm_buf), "%d", result.tile.bm);
            std::snprintf(bn_buf, sizeof(bn_buf), "%d", result.tile.bn);
            std::snprintf(bk_buf, sizeof(bk_buf), "%d", result.tile.bk);
        }

        std::printf("%-6s %-6s %-6s %-6s %-10.4f %-10.2f %-10.6f %-8s\n",
                    result.kernel_name,
                    bm_buf,
                    bn_buf,
                    bk_buf,
                    result.avg_ms,
                    result.tflops,
                    result.stats.max_abs_err,
                    result.valid ? "yes" : "no");
    }

    if (best.valid) {
        if (std::strcmp(best.kernel_name, "cublas") == 0) {
            std::printf("\nBest valid case: cublas, avg_ms=%.4f, TFLOPS=%.2f\n",
                        best.avg_ms,
                        best.tflops);
        } else {
            std::printf("\nBest valid case: %s BM=%d BN=%d BK=%d, avg_ms=%.4f, TFLOPS=%.2f\n",
                        best.kernel_name,
                        best.tile.bm,
                        best.tile.bn,
                        best.tile.bk,
                        best.avg_ms,
                        best.tflops);
        }
    }

    if (best_custom.valid) {
        std::printf("Best custom config: %s BM=%d BN=%d BK=%d, avg_ms=%.4f, TFLOPS=%.2f\n",
                    best_custom.kernel_name,
                    best_custom.tile.bm,
                    best_custom.tile.bn,
                    best_custom.tile.bk,
                    best_custom.avg_ms,
                    best_custom.tflops);
    } else {
        std::printf("No valid custom config found.\n");
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_custom));
    CHECK_CUDA(cudaFree(dC_cublas));

    return 0;
}
