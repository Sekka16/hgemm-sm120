#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

constexpr int kWarpSize = 32;

__device__ __forceinline__ void cp_async_16b(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

template <
    int BM,
    int BK,
    int BLOCK_SIZE,
    int COPY_BYTES = 16
>
__device__ __forceinline__ void load_a_stage(
    half (&s_a)[BM][BK],
    const half* A,
    int by,
    int K,
    int tid,
    int ko
) {
    constexpr int ELEMS_PER_COPY = COPY_BYTES / sizeof(half);   // 8
    constexpr int THREADS_PER_ROW = BK / ELEMS_PER_COPY;

    static_assert(BK % ELEMS_PER_COPY == 0);
    static_assert(BLOCK_SIZE % THREADS_PER_ROW == 0);

    const int gmem_m_base = by * BM;
    const int smem_m_init = tid / THREADS_PER_ROW;
    const int smem_k_init = (tid % THREADS_PER_ROW) * ELEMS_PER_COPY;
    const int stride_m = BLOCK_SIZE / THREADS_PER_ROW;

    #pragma unroll
    for (int step = 0; step < BM; step += stride_m) {
        const int smem_m = smem_m_init + step;
        const int smem_k = smem_k_init;

        const int gmem_m = gmem_m_base + smem_m;
        const int gmem_k = ko + smem_k;

        cp_async_16b(&s_a[smem_m][smem_k], &A[gmem_m * K + gmem_k]);
    }
}

template <
    int BK,
    int BN,
    int BLOCK_SIZE,
    int COPY_BYTES = 16,
    int PADDING = 8
>
__device__ __forceinline__ void load_b_stage(
    half (&s_b)[BK][BN + PADDING],
    const half* B,
    int bx,
    int N,
    int tid,
    int ko
) {
    constexpr int ELEMS_PER_COPY = COPY_BYTES / sizeof(half);   // 8
    constexpr int THREADS_PER_ROW = BN / ELEMS_PER_COPY;

    static_assert(BN % ELEMS_PER_COPY == 0);
    static_assert(BLOCK_SIZE % THREADS_PER_ROW == 0);

    const int gmem_n_base = bx * BN;
    const int smem_k_init = tid / THREADS_PER_ROW;
    const int smem_n_init = (tid % THREADS_PER_ROW) * ELEMS_PER_COPY;
    const int stride_k = BLOCK_SIZE / THREADS_PER_ROW;

    #pragma unroll
    for (int step = 0; step < BK; step += stride_k) {
        const int smem_k = smem_k_init + step;
        const int smem_n = smem_n_init;

        const int gmem_k = ko + smem_k;
        const int gmem_n = gmem_n_base + smem_n;

        cp_async_16b(&s_b[smem_k][smem_n], &B[gmem_k * N + gmem_n]);
    }
}

template <
    int BM,
    int BN,
    int BK,
    int BLOCK_SIZE,
    int COPY_BYTES = 16
>
__global__ void hgemm_wmma_kernel(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    constexpr int WARP_TILE_M = 64;
    constexpr int WARP_TILE_N = 64;

    constexpr int WARPS_M = BM / WARP_TILE_M;
    constexpr int WARPS_N = BN / WARP_TILE_N;

    constexpr int FRAG_M = WARP_TILE_M / WMMA_M;   // 4
    constexpr int FRAG_N = WARP_TILE_N / WMMA_N;   // 4
    constexpr int FRAG_K = BK / WMMA_K;            // 2

    static_assert(BM % WARP_TILE_M == 0);
    static_assert(BN % WARP_TILE_N == 0);
    static_assert(BK % WMMA_K == 0);
    static_assert(BLOCK_SIZE == WARPS_M * WARPS_N * kWarpSize);

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int wid = tid / kWarpSize;

    const int warp_m = wid / WARPS_N;
    const int warp_n = wid % WARPS_N;

    constexpr int PADDING = 8;

    extern __shared__ half smem[];

    half* s_a_ptr = smem;
    half* s_b_ptr = s_a_ptr + 2 * BM * BK;

    half (*s_a)[BM][BK] = reinterpret_cast<half (*)[BM][BK]>(s_a_ptr);
    half (*s_b)[BK][BN + PADDING] = reinterpret_cast<half (*)[BK][BN + PADDING]>(s_b_ptr);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[FRAG_M][FRAG_K];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[FRAG_K][FRAG_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[FRAG_M][FRAG_N];

    #pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    int stage = 0;

    load_a_stage<BM, BK, BLOCK_SIZE, COPY_BYTES>(s_a[stage], A, by, K, tid, 0);
    load_b_stage<BK, BN, BLOCK_SIZE, COPY_BYTES, PADDING>(s_b[stage], B, bx, N, tid, 0);

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    for (int ko = 0; ko < K; ko += BK) {
        const int next_stage = stage ^ 1;
        const int ko_next = ko + BK;

        if (ko_next < K) {
            load_a_stage<BM, BK, BLOCK_SIZE, COPY_BYTES>(s_a[next_stage], A, by, K, tid, ko_next);
            load_b_stage<BK, BN, BLOCK_SIZE, COPY_BYTES, PADDING>(s_b[next_stage], B, bx, N, tid, ko_next);
            cp_async_commit();
        }

        #pragma unroll
        for (int i = 0; i < FRAG_M; i++) {
            #pragma unroll
            for (int j = 0; j < FRAG_K; j++) {
                const half* a_ptr = &s_a[stage][warp_m * WARP_TILE_M + i * WMMA_M][j * WMMA_K];
                wmma::load_matrix_sync(frag_a[i][j], a_ptr, BK);
            }
        }

        #pragma unroll
        for (int i = 0; i < FRAG_K; i++) {
            #pragma unroll
            for (int j = 0; j < FRAG_N; j++) {
                const half* b_ptr = &s_b[stage][i * WMMA_K][warp_n * WARP_TILE_N + j * WMMA_N];
                wmma::load_matrix_sync(frag_b[i][j], b_ptr, BN + PADDING);
            }
        }

        #pragma unroll
        for (int i = 0; i < FRAG_M; i++) {
            #pragma unroll
            for (int j = 0; j < FRAG_N; j++) {
                #pragma unroll
                for (int k_frag = 0; k_frag < FRAG_K; k_frag++) {
                    wmma::mma_sync(frag_c[i][j], frag_a[i][k_frag], frag_b[k_frag][j], frag_c[i][j]);
                }
            }
        }

        if (ko_next < K) {
            cp_async_wait_all();
            __syncthreads();
        }

        stage = next_stage;
    }

    const int c_row_base = by * BM + warp_m * WARP_TILE_M;
    const int c_col_base = bx * BN + warp_n * WARP_TILE_N;

    #pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            float* c_ptr = &C[(c_row_base + i * WMMA_M) * N + (c_col_base + j * WMMA_N)];
            wmma::store_matrix_sync(c_ptr, frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}

template <int BM, int BN, int BK, int BLOCK_SIZE>
void launch_hgemm_wmma(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K
) {
    constexpr int PADDING = 8;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    size_t smem_size =
        2 * BM * BK * sizeof(half) +
        2 * BK * (BN + PADDING) * sizeof(half);

    // 👉 关键：允许超过48KB
    cudaFuncSetAttribute(
        hgemm_wmma_kernel<BM, BN, BK, BLOCK_SIZE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        50176  // 或者 50176 也够
    );

    hgemm_wmma_kernel<BM, BN, BK, BLOCK_SIZE>
        <<<grid, block, smem_size>>>(A, B, C, M, N, K);
}

void dispatch_hgemm_wmma(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K,
    int bm,
    int bn
) {
    if (bm == 128 && bn == 128) {
        launch_hgemm_wmma<128, 128, 32, 128>(A, B, C, M, N, K);
    } else if (bm == 128 && bn == 256) {
        launch_hgemm_wmma<128, 256, 32, 256>(A, B, C, M, N, K);
    } else {
        printf("Unsupported tile shape: BM=%d, BN=%d\n", bm, bn);
    }
}