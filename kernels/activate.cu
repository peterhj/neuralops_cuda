#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void activate_rect_fwd_kernel(
    const float *in_act,
    uint32_t dim,
    float *out_act)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_act[idx];
    out_act[idx] = x * (x > 0.0f);
  }
}

extern "C" void neuralops_cuda_activate_rect_fwd(
    const float *in_act,
    size_t dim,
    float *out_act,
    cudaStream_t stream)
{
  activate_rect_fwd_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      in_act, dim, out_act);
}

__global__ void activate_rect_bwd_kernel(
    const float *in_act,
    uint32_t dim,
    const float *out_delta,
    float *in_delta)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_act[idx];
    in_delta[idx] = out_delta[idx] * (x > 0.0f);
  }
}

extern "C" void neuralops_cuda_activate_rect_bwd(
    const float *in_act,
    size_t dim,
    const float *out_delta,
    float *in_delta,
    cudaStream_t stream)
{
  activate_rect_bwd_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      in_act, dim, out_delta, in_delta);
}

__global__ void activate_rect_bwd2_kernel(
    const float *in_act,
    uint32_t dim,
    const float *out_delta2,
    float *in_delta2)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_act[idx];
    in_delta2[idx] = out_delta2[idx] * (x > 0.0f);
  }
}

extern "C" void neuralops_cuda_activate_rect_bwd2(
    const float *in_act,
    size_t dim,
    const float *out_delta2,
    float *in_delta2,
    cudaStream_t stream)
{
  activate_rect_bwd2_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      in_act, dim, out_delta2, in_delta2);
}

__global__ void activate_rect_rfwd_kernel(
    const float *in_val,
    uint32_t dim,
    const float *in_r_val,
    float *out_r_val)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_val[idx];
    out_r_val[idx] = in_r_val[idx] * (x > 0.0f);
  }
}

extern "C" void neuralops_cuda_activate_rect_rfwd(
    const float *in_val,
    size_t dim,
    const float *in_r_val,
    float *out_r_val,
    cudaStream_t stream)
{
  activate_rect_rfwd_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      in_val, dim, in_r_val, out_r_val);
}

__global__ void activate_rect_rbwd_kernel(
    const float *in_val,
    uint32_t dim,
    const float *out_r_grad,
    float *in_r_grad)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_val[idx];
    in_r_grad[idx] = out_r_grad[idx] * (x > 0.0f);
  }
}

extern "C" void neuralops_cuda_activate_rect_rbwd(
    const float *in_val,
    size_t dim,
    const float *out_r_grad,
    float *in_r_grad,
    cudaStream_t stream)
{
  activate_rect_rbwd_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      in_val, dim, out_r_grad, in_r_grad);
}

__global__ void activate_leakrect_fwd_kernel(
    const float *in_act,
    uint32_t dim,
    float *out_act,
    float neg_slope)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_act[idx];
    int mask = x > 0.0f;
    out_act[idx] = x * (neg_slope * (1 - mask) + mask);
  }
}

extern "C" void neuralops_cuda_activate_leakrect_fwd(
    const float *in_act,
    size_t dim,
    float *out_act,
    float neg_slope,
    cudaStream_t stream)
{
  activate_leakrect_fwd_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      in_act, dim, out_act, neg_slope);
}

__global__ void activate_leakrect_bwd_kernel(
    const float *in_act,
    uint32_t dim,
    const float *out_delta,
    float *in_delta,
    float neg_slope)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_act[idx];
    int mask = x > 0.0f;
    float dy = out_delta[idx];
    in_delta[idx] = dy * (neg_slope * (1 - mask) + mask);
  }
}

extern "C" void neuralops_cuda_activate_leakrect_bwd(
    const float *in_act,
    size_t dim,
    const float *out_delta,
    float *in_delta,
    float neg_slope,
    cudaStream_t stream)
{
  activate_leakrect_bwd_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      in_act, dim, out_delta, in_delta, neg_slope);
}

__global__ void activate_logistic_fwd_kernel(
    const float *in_buf,
    uint32_t dim,
    float *out_buf)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_buf[idx];
    out_buf[idx] = 1.0f / (1.0f + expf(-x));
  }
}

extern "C" void neuralops_cuda_activate_logistic_fwd(
    const float *in_buf,
    size_t dim,
    float *out_buf,
    cudaStream_t stream)
{
  activate_logistic_fwd_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      in_buf, dim, out_buf);
}

__global__ void activate_logistic_bwd_kernel(
    const float *in_buf,
    uint32_t dim,
    const float *out_delta,
    float *in_delta)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_buf[idx];
    float y = 1.0f / (1.0f + expf(-x));
    in_delta[idx] = y * (1.0f - y) * out_delta[idx];
  }
}

extern "C" void neuralops_cuda_activate_logistic_bwd(
    const float *in_buf,
    size_t dim,
    const float *out_delta,
    float *in_delta,
    cudaStream_t stream)
{
  activate_logistic_bwd_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      in_buf, dim, out_delta, in_delta);
}
