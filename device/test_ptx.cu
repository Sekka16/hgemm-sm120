#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err__));                                      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void div_kernel(int* warp_out, int* lane_out) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned warp_id = tid / 32;
  unsigned lane_id = tid % 32;

  warp_out[tid] = static_cast<int>(warp_id);
  lane_out[tid] = static_cast<int>(lane_id);
}

__global__ void shift_kernel(int* warp_out, int* lane_out) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned warp_id = tid >> 5;
  unsigned lane_id = tid & 31u;

  warp_out[tid] = static_cast<int>(warp_id);
  lane_out[tid] = static_cast<int>(lane_id);
}

int main() {
  constexpr int N = 256;
  const size_t bytes = static_cast<size_t>(N) * sizeof(int);

  int *d_warp1 = nullptr, *d_lane1 = nullptr;
  int *d_warp2 = nullptr, *d_lane2 = nullptr;

  int h_warp1[N], h_lane1[N], h_warp2[N], h_lane2[N];

  CHECK_CUDA(cudaMalloc(&d_warp1, bytes));
  CHECK_CUDA(cudaMalloc(&d_lane1, bytes));
  CHECK_CUDA(cudaMalloc(&d_warp2, bytes));
  CHECK_CUDA(cudaMalloc(&d_lane2, bytes));

  div_kernel<<<1, N>>>(d_warp1, d_lane1);
  CHECK_CUDA(cudaGetLastError());

  shift_kernel<<<1, N>>>(d_warp2, d_lane2);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_warp1, d_warp1, bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_lane1, d_lane1, bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_warp2, d_warp2, bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_lane2, d_lane2, bytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < N; ++i) {
    if (h_warp1[i] != h_warp2[i] || h_lane1[i] != h_lane2[i]) {
      ok = false;
      printf("Mismatch at %d: div=(%d,%d), shift=(%d,%d)\n",
             i, h_warp1[i], h_lane1[i], h_warp2[i], h_lane2[i]);
      break;
    }
  }

  if (ok) {
    printf("Results match.\n");
    for (int i = 0; i < 8; ++i) {
      printf("tid=%d -> warp=%d lane=%d\n", i, h_warp1[i], h_lane1[i]);
    }
  }

  CHECK_CUDA(cudaFree(d_warp1));
  CHECK_CUDA(cudaFree(d_lane1));
  CHECK_CUDA(cudaFree(d_warp2));
  CHECK_CUDA(cudaFree(d_lane2));

  return 0;
}