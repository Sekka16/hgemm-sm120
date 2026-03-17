#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>

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

// -------------------- 初始化 --------------------
void init_matrix_half(std::vector<half>& mat, float scale = 1.0f, uint32_t seed = 12345) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = float_to_half_host(dist(rng));
    }
}

// -------------------- 结果对比 --------------------
void compare_results(const std::vector<float>& ref,
                     const std::vector<float>& out,
                     int M, int N) {
    double max_abs_err = 0.0;
    double avg_abs_err = 0.0;
    double max_rel_err = 0.0;
    int idx_max = -1;

    const int total = M * N;
    for (int i = 0; i < total; ++i) {
        const double a = static_cast<double>(ref[i]);
        const double b = static_cast<double>(out[i]);
        const double abs_err = std::abs(a - b);
        const double rel_err = abs_err / (std::abs(a) + 1e-6);

        avg_abs_err += abs_err;
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            idx_max = i;
        }
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }
    }

    avg_abs_err /= static_cast<double>(total);

    std::printf("max_abs_err = %.8f\n", max_abs_err);
    std::printf("avg_abs_err = %.8f\n", avg_abs_err);
    std::printf("max_rel_err = %.8f\n", max_rel_err);

    if (idx_max >= 0) {
        const int r = idx_max / N;
        const int c = idx_max % N;
        std::printf("worst at (%d, %d): ref = %.8f, out = %.8f\n",
                    r, c, ref[idx_max], out[idx_max]);
    }
}

// -------------------- 自定义 kernel launch --------------------
void launch_custom_hgemm(const half* dA,
                         const half* dB,
                         float* dC,
                         int M, int N, int K,
                         int tile_bm,
                         int tile_bn) {
    dispatch_hgemm_wmma(dA, dB, dC, M, N, K, tile_bm, tile_bn);
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

int main() {
    const int M = 8192;
    const int N = 8192;
    const int K = 8192;

    const int tile_bm = 128;
    const int tile_bn = 128;

    std::printf("Test shape: M=%d, N=%d, K=%d\n", M, N, K);
    std::printf("Tile shape: BM=%d, BN=%d\n", tile_bm, tile_bn);

    if (M % tile_bm != 0 || N % tile_bn != 0 || K % 32 != 0) {
        std::fprintf(stderr,
                     "Shape not aligned with current kernel assumptions. "
                     "Require M %% BM == 0, N %% BN == 0, K %% 32 == 0.\n");
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
    CHECK_CUDA(cudaMemset(dC_custom, 0, sizeof(float) * static_cast<size_t>(M) * N));
    launch_custom_hgemm(dA, dB, dC_custom, M, N, K, tile_bm, tile_bn);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemset(dC_cublas, 0, sizeof(float) * static_cast<size_t>(M) * N));
    launch_cublas_hgemm(handle, dA, dB, dC_cublas, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC_custom.data(), dC_custom,
                          sizeof(float) * static_cast<size_t>(M) * N,
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_cublas.data(), dC_cublas,
                          sizeof(float) * static_cast<size_t>(M) * N,
                          cudaMemcpyDeviceToHost));

    std::printf("\nCompare custom vs cuBLAS:\n");
    compare_results(hC_cublas, hC_custom, M, N);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_custom));
    CHECK_CUDA(cudaFree(dC_cublas));

    return 0;
}