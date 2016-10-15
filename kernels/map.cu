#include <cuda_runtime_api.h>

__global__ void batchmap_add_kernel(
    float *xs,
    int frame_len,
    int batch_size,
    float alpha,
    const float *scalars)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx % frame_len;
  int batch_idx = idx / frame_len;
  if ((i < frame_len) && (batch_idx < batch_size)) {
    float x = xs[idx] + alpha * scalars[batch_idx];
    xs[idx] = x;
  }
}

extern "C" void neuralops_cuda_batchmap_add(
    float *xs,
    size_t frame_len,
    size_t batch_size,
    float alpha,
    const float *scalars,
    cudaStream_t stream)
{
  int n = frame_len * batch_size;
  batchmap_add_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(xs, frame_len, batch_size, alpha, scalars);
}

__global__ void batchmap_div_kernel(
    float *xs,
    int frame_len,
    int batch_size,
    const float *scalars)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx % frame_len;
  int batch_idx = idx / frame_len;
  if ((i < frame_len) && (batch_idx < batch_size)) {
    float x = xs[idx] / scalars[batch_idx];
    xs[idx] = x;
  }
}

extern "C" void neuralops_cuda_batchmap_div(
    float *xs,
    size_t frame_len,
    size_t batch_size,
    const float *scalars,
    cudaStream_t stream)
{
  int n = frame_len * batch_size;
  batchmap_div_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(xs, frame_len, batch_size, scalars);
}
