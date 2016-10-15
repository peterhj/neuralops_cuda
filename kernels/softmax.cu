#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void softmax_nll_loss_fwd_kernel(
    const float *out_act,
    int dim,
    int batch_size,
    const uint32_t *label_cats,
    const float *weights,
    const float *targets,
    float *out_loss)
{
  int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_size) {
    int cat_i = label_cats[batch_idx];
    int idx = cat_i + batch_idx * dim;
    float x = -logf(out_act[idx]) * weights[batch_idx] * targets[batch_idx];
    out_loss[batch_idx] = x;
  }
}

extern "C" void neuralops_cuda_softmax_nll_loss_fwd(
    const float *out_act,
    size_t dim,
    size_t batch_size,
    const uint32_t *label_cats,
    const float *weights,
    const float *targets,
    float *out_loss,
    cudaStream_t stream)
{
  int n = batch_size;
  softmax_nll_loss_fwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act, dim, batch_size,
      label_cats,
      weights,
      targets,
      out_loss);
}

__global__ void softmax_nll_loss_bwd_kernel(
    const float *out_act,
    int dim,
    int batch_size,
    const uint32_t *label_cats,
    const float *weights,
    const float *targets,
    float *in_delta)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx % dim;
  int batch_idx = idx / dim;
  if ((i < dim) && (batch_idx < batch_size)) {
    int cat_i = label_cats[batch_idx];
    float dx = out_act[idx];
    if ((uint32_t)(i) == cat_i) {
      dx -= 1.0f;
    }
    dx *= weights[batch_idx] * targets[batch_idx];
    in_delta[idx] = dx;
  }
}

extern "C" void neuralops_cuda_softmax_nll_loss_bwd(
    const float *out_act,
    size_t dim,
    size_t batch_size,
    const uint32_t *label_cats,
    const float *weights,
    const float *targets,
    float *in_delta,
    cudaStream_t stream)
{
  int n = dim * batch_size;
  softmax_nll_loss_bwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act, dim, batch_size,
      label_cats,
      weights,
      targets,
      in_delta);
}
