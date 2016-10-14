#include <cuda_runtime_api.h>
//#include <stdint.h>

__global__ void activate_rect_fwd_kernel(
    const float *in_act,
    int dim,
    float *out_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_act[idx];
    if (x > 0.0f) {
      out_act[idx] = x;
    } else {
      out_act[idx] = 0.0f;
    }
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
    int dim,
    const float *out_delta,
    float *in_delta)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = in_act[idx];
    if (x > 0.0f) {
      in_delta[idx] = out_delta[idx];
    } else {
      in_delta[idx] = 0.0f;
    }
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
