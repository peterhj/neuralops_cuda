#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void logistic_nll_loss_fwd_kernel(
    const float *in_buf,
    uint32_t batch_size,
    const uint32_t *binary_labels,
    const float *weights,
    float *loss,
    float *pred)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < batch_size) {
    float x = in_buf[idx];
    float y = 1.0f / (1.0f + expf(-x));
    float y_target = (float)(binary_labels[idx] > 0);
    float ell = -weights[idx] * ((1.0f - y_target) * logf(1.0f - y) + y_target * logf(y));
    loss[idx] = ell;
    pred[idx] = y;
  }
}

extern "C" void neuralops_cuda_logistic_nll_loss_fwd(
    const float *in_buf,
    size_t batch_size,
    const uint32_t *binary_labels,
    const float *weights,
    float *loss,
    float *pred,
    cudaStream_t stream)
{
  logistic_nll_loss_fwd_kernel<<<(batch_size+1024-1)/1024, 1024, 0, stream>>>(
      in_buf, batch_size, binary_labels, weights, loss, pred);
}

__global__ void logistic_nll_loss_bwd_kernel(
    const float *in_buf,
    uint32_t batch_size,
    const uint32_t *binary_labels,
    const float *weights,
    float *in_delta)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < batch_size) {
    float x = in_buf[idx];
    float y = 1.0f / (1.0f + expf(-x));
    float y_target = (float)(binary_labels[idx] > 0);
    float dx = weights[idx] * (y - y_target);
    in_delta[idx] = dx;
  }
}

extern "C" void neuralops_cuda_logistic_nll_loss_bwd(
    const float *in_buf,
    size_t batch_size,
    const uint32_t *binary_labels,
    const float *weights,
    float *in_delta,
    cudaStream_t stream)
{
  logistic_nll_loss_bwd_kernel<<<(batch_size+1024-1)/1024, 1024, 0, stream>>>(
      in_buf, batch_size, binary_labels, weights, in_delta);
}
