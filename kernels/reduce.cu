#include <cuda_runtime_api.h>
#include <stdint.h>

#define OFFSET_BANK(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) / 32)); })

__global__ void batch_blockreduce_argmax_kernel(
    const float *xs,
    int len,
    int batch_size,
    float *x_max_block,
    uint32_t *x_argmax_block)
{
  __shared__ float cache[1024 + 32];
  __shared__ int cache_idx[1024 + 32];
  int tid = threadIdx.x;
  int block = blockIdx.x;
  int i = tid + block * len;
  if (tid < len && block < batch_size) {
    cache[OFFSET_BANK(tid)]     = xs[i];
    cache_idx[OFFSET_BANK(tid)] = tid;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid < len && block < batch_size) {
      if (tid % (2*s) == 0 && (tid + s) < len && cache[OFFSET_BANK(tid)] < cache[OFFSET_BANK(tid + s)]) {
        cache[OFFSET_BANK(tid)]     = cache[OFFSET_BANK(tid + s)];
        cache_idx[OFFSET_BANK(tid)] = cache_idx[OFFSET_BANK(tid + s)];
      }
    }
    __syncthreads();
  }
  if (tid < len && block < batch_size) {
    if (tid == 0) {
      x_max_block[block] = cache[0];
      if (x_argmax_block != NULL) {
        x_argmax_block[block] = cache_idx[0];
      }
    }
  }
}

extern "C" void neuralops_cuda_blockreduce_max_argmax(
    const float *xs,
    size_t len,
    size_t batch_size,
    float *xs_max,
    uint32_t *xs_argmax,
    cudaStream_t stream)
{
  // XXX: assert(len <= 1024);
  // FIXME(20151022): could make more efficient use of blocks but w/e.
  int n = batch_size * 1024;
  batch_blockreduce_argmax_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      xs, len, batch_size, xs_max, xs_argmax);
}

__global__ void batch_blockreduce_sum_kernel(
    const float *xs,
    int len,
    int batch_size,
    float *xs_sum,
    float alpha)
{
  __shared__ float cache[1024 + 32];
  int tid = threadIdx.x;
  int block = blockIdx.x;
  int i = tid + block * len;
  if (tid < len && block < batch_size) {
    cache[OFFSET_BANK(tid)] = xs[i];
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid < len && block < batch_size) {
      if (tid % (2*s) == 0 && (tid + s) < len) {
        cache[OFFSET_BANK(tid)] += cache[OFFSET_BANK(tid + s)];
      }
    }
    __syncthreads();
  }
  if (tid < len && block < batch_size) {
    if (tid == 0) {
      if (alpha != 0.0f) {
        float xs_sum_0 = xs_sum[block];
        xs_sum[block] = alpha * xs_sum_0 + cache[0];
      } else {
        xs_sum[block] = cache[0];
      }
    }
  }
}

extern "C" void neuralops_cuda_blockreduce_sum(
    const float *xs,
    size_t len,
    size_t batch_size,
    float alpha,
    float *xs_sum,
    cudaStream_t stream)
{
  // XXX: assert(len <= 1024);
  // FIXME(20151022): could make more efficient use of blocks but w/e.
  int n = batch_size * 1024;
  batch_blockreduce_sum_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      xs, len, batch_size, xs_sum, alpha);
}
