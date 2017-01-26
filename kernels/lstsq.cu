#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void lst_sq_fwd_kernel(
    const float *x,
    uint32_t batch_sz,
    const float *targets,
    const float *weights,
    float *loss)
{
  uint32_t batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_sz) {
    float dx = x[batch_idx] - targets[batch_idx];
    loss[batch_idx] = 0.5f * weights[batch_idx] * dx * dx;
  }
}

extern "C" void neuralops_cuda_lst_sq_fwd(
    const float *x,
    size_t batch_sz,
    const float *targets,
    const float *weights,
    float *loss,
    cudaStream_t stream)
{
  lst_sq_fwd_kernel<<<(batch_sz+1024-1)/1024, 1024, 0, stream>>>(
      x, batch_sz, targets, weights, loss);
}

__global__ void lst_sq_bwd_kernel(
    const float *x,
    uint32_t batch_sz,
    const float *targets,
    const float *weights,
    float *grad)
{
  uint32_t batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_sz) {
    grad[batch_idx] = weights[batch_idx] * (x[batch_idx] - targets[batch_idx]);
  }
}

extern "C" void neuralops_cuda_lst_sq_bwd(
    const float *x,
    size_t batch_sz,
    const float *targets,
    const float *weights,
    float *grad,
    cudaStream_t stream)
{
  lst_sq_bwd_kernel<<<(batch_sz+1024-1)/1024, 1024, 0, stream>>>(
      x, batch_sz, targets, weights, grad);
}

__global__ void lst_sq_rfwd_kernel(
    const float *x,
    uint32_t batch_sz,
    const float *rx,
    const float *targets,
    const float *jac_targ,
    float *r_loss)
{
  uint32_t batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_sz) {
    r_loss[batch_idx] = jac_targ[batch_idx] * (x[batch_idx] - targets[batch_idx]) * rx[batch_idx];
  }
}

extern "C" void neuralops_cuda_lst_sq_rfwd(
    const float *x,
    size_t batch_sz,
    const float *rx,
    const float *targets,
    const float *jac_targ,
    float *r_loss,
    cudaStream_t stream)
{
  lst_sq_rfwd_kernel<<<(batch_sz+1024-1)/1024, 1024, 0, stream>>>(
      x, batch_sz, rx, targets, jac_targ, r_loss);
}

__global__ void ind_lst_sq_fwd_kernel(
    const float *x,
    uint32_t dim,
    uint32_t batch_sz,
    const float *targets,
    const uint32_t *labels,
    const float *weights,
    float *loss)
{
  uint32_t batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_sz) {
    uint32_t label_k = labels[batch_idx];
    if (label_k < dim) {
      uint32_t idx = label_k + dim * batch_idx;
      float dx = x[idx] - targets[batch_idx];
      //loss[batch_idx] = 0.5f * weights[batch_idx] * dx * dx;
      loss[batch_idx] = 0.5f * dx * dx;
    } else {
      loss[batch_idx] = 0.0f;
    }
  }
}

extern "C" void neuralops_cuda_ind_lst_sq_fwd(
    const float *x,
    size_t dim,
    size_t batch_sz,
    const float *targets,
    const uint32_t *labels,
    const float *weights,
    float *loss,
    cudaStream_t stream)
{
  ind_lst_sq_fwd_kernel<<<(batch_sz+1024-1)/1024, 1024, 0, stream>>>(
      x, dim, batch_sz, targets, labels, weights, loss);
}

__global__ void ind_lst_sq_bwd_kernel(
    const float *x,
    uint32_t dim,
    uint32_t batch_sz,
    const float *targets,
    const uint32_t *labels,
    const float *weights,
    float *grad)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t k = idx % dim;
  uint32_t batch_idx = idx / dim;
  if (k < dim && batch_idx < batch_sz) {
    if (k == labels[batch_idx]) {
      grad[idx] = x[idx] - targets[batch_idx];
    } else {
      grad[idx] = 0.0f;
    }
  }
}

extern "C" void neuralops_cuda_ind_lst_sq_bwd(
    const float *x,
    size_t dim,
    size_t batch_sz,
    const float *targets,
    const uint32_t *labels,
    const float *weights,
    float *grad,
    cudaStream_t stream)
{
  uint32_t n = dim * batch_sz;
  ind_lst_sq_bwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      x, dim, batch_sz, targets, labels, weights, grad);
}
