#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void gaussian_kl_loss_fwd_kernel(
    const float *mean,
    uint32_t batch_sz,
    const float *target_mean,
    float var,
    float target_var,
    float *loss)
{
  uint32_t batch_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (batch_idx < batch_sz) {
    float x = mean[batch_idx];
    float t = target_mean[batch_idx];
    loss[batch_idx] = 0.5f * ((target_var + (x - t) * (x - t)) / var - logf(target_var) + logf(var) - 1.0f);
  }
}

extern "C" void neuralops_cuda_gaussian_kl_loss_fwd(
    const float *mean,
    uint32_t batch_sz,
    const float *target_mean,
    float var,
    float target_var,
    float *loss,
    cudaStream_t stream)
{
  gaussian_kl_loss_fwd_kernel<<<(batch_sz+1024-1)/1024, 1024, 0, stream>>>(
      mean, batch_sz, target_mean, var, target_var, loss);
}
