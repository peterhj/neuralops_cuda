#include <cuda_runtime_api.h>
#include <stdint.h>

#define OFFSET_BANK(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) / 32)); })

__global__ void softmax_kl_loss_fwd_kernel(
    const float *ys,
    uint32_t dim,
    uint32_t batch_sz,
    const float *targets,
    float *loss)
{
  __shared__ float cache[1024 + 32];
  uint32_t j = threadIdx.x;
  uint32_t batch_idx = blockIdx.x;
  uint32_t idx = j + dim * batch_idx;
  if (j < dim && batch_idx < batch_sz) {
    float t = targets[idx];
    float y = ys[idx];
    float kl_j = t * (logf(t) - logf(y));
    cache[OFFSET_BANK(j)] = kl_j;
  } else {
    cache[OFFSET_BANK(j)] = 0.0f;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (j < dim && batch_idx < batch_sz) {
      if (j % (2*s) == 0 && (j + s) < dim && cache[OFFSET_BANK(j)] < cache[OFFSET_BANK(j + s)]) {
        cache[OFFSET_BANK(j)] = cache[OFFSET_BANK(j + s)];
      }
    }
    __syncthreads();
  }
  if (j < dim && batch_idx < batch_sz) {
    if (j == 0) {
      loss[batch_idx] = cache[0];
    }
  }
}

extern "C" void neuralops_cuda_softmax_kl_loss_fwd(
    const float *ys,
    uint32_t dim,
    uint32_t batch_sz,
    const float *targets,
    float *loss,
    cudaStream_t stream)
{
  //assert(dim <= 1024);
  softmax_kl_loss_fwd_kernel<<<batch_sz, 1024, 0, stream>>>(
      ys, dim, batch_sz, targets, loss);
}

__global__ void softmax_kl_loss_bwd_kernel(
    const float *ys,
    uint32_t dim,
    uint32_t batch_sz,
    const float *targets,
    float *grad)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t j = idx % dim;
  uint32_t batch_idx = idx / dim;
  if (j < dim && batch_idx < batch_sz) {
    float t = targets[idx];
    float y = ys[idx];
    grad[idx] = y - t;
  }
}

extern "C" void neuralops_cuda_softmax_kl_loss_bwd(
    const float *ys,
    uint32_t dim,
    uint32_t batch_sz,
    const float *targets,
    float *grad,
    cudaStream_t stream)
{
  uint32_t n = dim * batch_sz;
  softmax_kl_loss_bwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      ys, dim, batch_sz, targets, grad);
}

__global__ void softmax_kl_loss_rfwd_kernel(
    const float *ys,
    uint32_t dim,
    uint32_t batch_sz,
    const float *r_xs,
    const float *r_mean,
    const float *targets,
    float *r_loss,
    float *r_grad)
{
  __shared__ float cache[1024 + 32];
  uint32_t j = threadIdx.x;
  uint32_t batch_idx = blockIdx.x;
  uint32_t idx = j + dim * batch_idx;
  if (j < dim && batch_idx < batch_sz) {
    float r_yp_i = r_xs[idx] - r_mean[batch_idx];
    r_grad[idx] = ys[idx] * r_yp_i;
    cache[OFFSET_BANK(j)] = -targets[idx] * r_yp_i;
  } else {
    cache[OFFSET_BANK(j)] = 0.0f;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (j < dim && batch_idx < batch_sz) {
      if (j % (2*s) == 0 && (j + s) < dim && cache[OFFSET_BANK(j)] < cache[OFFSET_BANK(j + s)]) {
        cache[OFFSET_BANK(j)] = cache[OFFSET_BANK(j + s)];
      }
    }
    __syncthreads();
  }
  if (j < dim && batch_idx < batch_sz) {
    if (j == 0) {
      r_loss[batch_idx] = cache[0];
    }
  }
}

extern "C" void neuralops_cuda_softmax_kl_loss_rfwd(
    const float *ys,
    uint32_t dim,
    uint32_t batch_sz,
    const float *r_xs,
    const float *r_mean,
    const float *targets,
    float *r_loss,
    float *r_grad,
    cudaStream_t stream)
{
  //assert(dim <= 1024);
  softmax_kl_loss_rfwd_kernel<<<batch_sz, 1024, 0, stream>>>(
      ys, dim, batch_sz, r_xs, r_mean, targets, r_loss, r_grad);
}

__global__ void softmax_nll_loss_fwd_kernel(
    const float *out_act,
    int dim,
    int batch_size,
    const uint32_t *label_cats,
    const float *weights,
    //const float *targets,
    float *out_loss)
{
  int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_size) {
    int cat_i = label_cats[batch_idx];
    int idx = cat_i + batch_idx * dim;
    float x = -logf(out_act[idx]) * weights[batch_idx];
    out_loss[batch_idx] = x;
  }
}

extern "C" void neuralops_cuda_softmax_nll_loss_fwd(
    const float *out_act,
    size_t dim,
    size_t batch_size,
    const uint32_t *label_cats,
    const float *weights,
    float *out_loss,
    cudaStream_t stream)
{
  int n = batch_size;
  softmax_nll_loss_fwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act, dim, batch_size,
      label_cats,
      weights,
      out_loss);
}

__global__ void softmax_nll_loss_bwd_kernel(
    const float *out_act,
    int dim,
    int batch_size,
    const uint32_t *label_cats,
    const float *weights,
    const float *jac_targ,
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
    dx *= weights[batch_idx] * jac_targ[batch_idx];
    in_delta[idx] = dx;
  }
}

extern "C" void neuralops_cuda_softmax_nll_loss_bwd(
    const float *out_act,
    size_t dim,
    size_t batch_size,
    const uint32_t *label_cats,
    const float *weights,
    const float *jac_targ,
    float *in_delta,
    cudaStream_t stream)
{
  int n = dim * batch_size;
  softmax_nll_loss_bwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act, dim, batch_size,
      label_cats,
      weights,
      jac_targ,
      in_delta);
}

__global__ void softmax_nll_loss_bwd2_kernel(
    const float *out_act,
    uint32_t dim,
    uint32_t batch_size,
    const uint32_t *label_cats,
    const float *weights,
    const float *jac_targ,
    float *in_delta2)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t i = idx % dim;
  uint32_t batch_idx = idx / dim;
  if ((i < dim) && (batch_idx < batch_size)) {
    uint32_t cat_i = label_cats[batch_idx];
    float y = out_act[idx];
    float dx2 = y;
    if ((uint32_t)(i) == cat_i) {
      dx2 -= 1.0f;
    }
    dx2 *= (1.0f - 2.0f * y) * weights[batch_idx] * jac_targ[batch_idx];
    in_delta2[idx] = dx2;
  }
}

extern "C" void neuralops_cuda_softmax_nll_loss_bwd2(
    const float *out_act,
    size_t dim,
    size_t batch_size,
    const uint32_t *label_cats,
    const float *weights,
    const float *jac_targ,
    float *in_delta2,
    cudaStream_t stream)
{
  int n = dim * batch_size;
  softmax_nll_loss_bwd2_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act,
      dim,
      batch_size,
      label_cats,
      weights,
      jac_targ,
      in_delta2);
}

__global__ void softmax_nll_loss_rfwd_kernel(
    const float *out_r_val,
    uint32_t dim,
    uint32_t batch_size,
    const float *r_mean,
    const uint32_t *labels,
    float *r_loss)
{
  uint32_t batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_size) {
    uint32_t cat_i = labels[batch_idx];
    uint32_t idx = cat_i + batch_idx * dim;
    float r_y_i = out_r_val[idx] - r_mean[batch_idx];
    r_loss[batch_idx] = -r_y_i;
  }
}

extern "C" void neuralops_cuda_softmax_nll_loss_rfwd(
    const float *out_r_val,
    size_t dim,
    size_t batch_size,
    const float *r_mean,
    const uint32_t *labels,
    float *r_loss,
    cudaStream_t stream)
{
  uint32_t n = batch_size;
  softmax_nll_loss_rfwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_r_val,
      dim,
      batch_size,
      r_mean,
      labels,
      r_loss);
}
