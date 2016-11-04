#include <cuda_runtime_api.h>

__global__ void linear_bias_fwd_kernel(
    const float *in_buf,
    int dim,
    int batch_size,
    const float *bias,
    float *out_buf)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int k = idx % dim;
  int batch_idx = idx / dim;
  if (k < dim && batch_idx < batch_size) {
    out_buf[idx] = in_buf[idx] + bias[k];
  }
}

extern "C" void neuralops_cuda_linear_bias_fwd(
    const float *in_buf,
    size_t dim,
    size_t batch_size,
    const float *bias,
    float *out_buf,
    cudaStream_t stream)
{
  int n = dim * batch_size;
  linear_bias_fwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_buf, dim, batch_size, bias, out_buf);
}

extern "C" void neuralops_cuda_linear_bias_fwd_inplace(
    float *out_buf,
    size_t dim,
    size_t batch_size,
    const float *bias,
    cudaStream_t stream)
{
  int n = dim * batch_size;
  linear_bias_fwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_buf, dim, batch_size, bias, out_buf);
}

__global__ void linear_bias_bwd_kernel(
    const float *out_grad,
    int dim,
    int batch_size,
    float *in_grad)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int k = idx % dim;
  int batch_idx = idx / dim;
  if (k < dim && batch_idx < batch_size) {
    atomicAdd(&in_grad[k], out_grad[idx]);
  }
}

extern "C" void neuralops_cuda_linear_bias_bwd(
    const float *out_grad,
    size_t dim,
    size_t batch_size,
    float *in_grad,
    cudaStream_t stream)
{
  int n = dim * batch_size;
  linear_bias_bwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_grad, dim, batch_size, in_grad);
}
