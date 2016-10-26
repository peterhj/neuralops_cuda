#include <cuda_runtime_api.h>

__global__ void clamp_kernel(
    float *y,
    int dim,
    float clamp_lo,
    float clamp_hi)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y_i = y[idx];
    if (y_i < clamp_lo) {
      y[idx] = clamp_lo;
    } else if (y_i > clamp_hi) {
      y[idx] = clamp_hi;
    }
  }
}

extern "C" void neuralops_cuda_clamp(
    float *y,
    size_t dim,
    float clamp_lo,
    float clamp_hi,
    cudaStream_t stream)
{
  clamp_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      y, dim, clamp_lo, clamp_hi);
}
