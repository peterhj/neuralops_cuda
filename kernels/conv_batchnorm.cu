#include <cuda_runtime_api.h>

#define OFFSET_BANK(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) / 32)); })

__global__ void conv_diag_affine_white_var_fwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *mean,
    const float *var,
    float epsilon,
    float *out_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int u = idx % spatial_dim;
  int c = (idx / spatial_dim) % num_channels;
  int batch_idx = idx / (spatial_dim * num_channels);
  if (u < spatial_dim && c < num_channels && batch_idx < batch_size) {
    float m = mean[c];
    float v = var[c];
    float y = (in_act[idx] - m) * rsqrtf(v + epsilon);
    out_act[idx] = y;
  }
}

extern "C" void neuralops_cuda_conv2d_whiten_fwd(
    const float *in_act,
    size_t spatial_dim,
    size_t num_channels,
    size_t batch_size,
    const float *mean,
    const float *var,
    float epsilon,
    float *out_act,
    cudaStream_t stream)
{
  int n = spatial_dim * num_channels * batch_size;
  conv_diag_affine_white_var_fwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, mean, var, epsilon, out_act);
}

__global__ void estimate_conv_mean_fast2_batch_kernel(
    const float *src,
    int spatial_dim,
    int num_channels,
    int batch_size,
    float *mean)
{
  __shared__ float mean_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  //int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int u0 = ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float y = 0.0f;
    /*int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      y += src[i];
    }*/
    int i0 = warp_idx + u0 + c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int i_limit = i0 + min(spatial_dim - warp_idx - u0, 16*32);
    for (int v = 0; v < 16*32; v += 32) {
      int i = i0 + v;
      if (i < i_limit) {
        y += src[i];
      }
    }
    mean_cache[bank_idx] = y;
  } else {
    mean_cache[bank_idx] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      mean_cache[bank_idx] += mean_cache[bank_idx+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      mean_cache[bank_idx] += mean_cache[bank_idx+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      mean_cache[bank_idx] += mean_cache[bank_idx+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      mean_cache[bank_idx] += mean_cache[bank_idx+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float y = (mean_cache[bank_idx] + mean_cache[bank_idx+16]) / ((float)(spatial_dim) * (float)(batch_size));
      atomicAdd(&mean[c], y);
    }
  }
}

extern "C" void neuralops_cuda_conv2d_mean_fwd(
    const float *src,
    size_t spatial_dim,
    size_t num_channels,
    size_t batch_size,
    float *mean,
    cudaStream_t stream)
{
  //int n = ((spatial_dim+32-1)/32) * channels * batch_size;
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int n = 32 * num_channels * block_spatial_dim * batch_size;
  estimate_conv_mean_fast2_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, spatial_dim, num_channels, batch_size, mean);
}

__global__ void estimate_conv_var_fast2_batch_kernel(
    const float *src,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *mean,
    float *var)
{
  __shared__ float var_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  //int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int u0 = ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float mean_c = mean[c];
    float y = 0.0f;
    /*int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float delta = src[i] - mean_c;
      y += delta * delta;
    }*/
    int i0 = warp_idx + u0 + c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int i_limit = i0 + min(spatial_dim - warp_idx - u0, 16*32);
    for (int v = 0; v < 16*32; v += 32) {
      int i = i0 + v;
      if (i < i_limit) {
        float delta = src[i] - mean_c;
        y += delta * delta;
      }
    }
    var_cache[bank_idx] = y;
  } else {
    var_cache[bank_idx] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      var_cache[bank_idx] += var_cache[bank_idx+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      var_cache[bank_idx] += var_cache[bank_idx+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      var_cache[bank_idx] += var_cache[bank_idx+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      var_cache[bank_idx] += var_cache[bank_idx+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float y = (var_cache[bank_idx] + var_cache[bank_idx+16]) / ((float)(spatial_dim-1) * (float)(batch_size-1));
      atomicAdd(&var[c], y);
    }
  }
}

extern "C" void neuralops_cuda_conv2d_var_fwd(
    const float *src,
    size_t spatial_dim,
    size_t num_channels,
    size_t batch_size,
    const float *mean,
    float *var,
    cudaStream_t stream)
{
  //int n = ((spatial_dim+32-1)/32) * channels * batch_size;
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int n = 32 * num_channels * block_spatial_dim * batch_size;
  estimate_conv_var_fast2_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, spatial_dim, num_channels, batch_size, mean, var);
}

__global__ void conv_bnorm_bwd_var_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *mean,
    const float *var,
    float epsilon,
    float *var_grad)
{
  __shared__ float d_sigma_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float mu = mean[c];
    float sigma = var[c];
    float inv_sqrt_sigma = rsqrtf(sigma + epsilon);
    float d_sigma = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      d_sigma += out_delta[i] * -0.5f * inv_sqrt_sigma / (sigma + epsilon) * (in_act[i] - mu);
    }
    d_sigma_cache[bank_idx] = d_sigma;
  } else {
    d_sigma_cache[bank_idx] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      d_sigma_cache[bank_idx] += d_sigma_cache[bank_idx+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      d_sigma_cache[bank_idx] += d_sigma_cache[bank_idx+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      d_sigma_cache[bank_idx] += d_sigma_cache[bank_idx+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      d_sigma_cache[bank_idx] += d_sigma_cache[bank_idx+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float d_sigma = d_sigma_cache[bank_idx] + d_sigma_cache[bank_idx+16];
      atomicAdd(&var_grad[c], d_sigma);
    }
  }
}

__global__ void conv_bnorm_bwd_mean_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *mean,
    const float *var,
    const float *var_grad,
    float epsilon,
    float *mean_grad)
{
  __shared__ float d_mu_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float inv_var_norm = 1.0f / ((float)(spatial_dim - 1) * (float)(batch_size - 1));
    float mu = mean[c];
    float sigma = var[c];
    float inv_sqrt_sigma = rsqrtf(sigma + epsilon);
    float d_sigma = var_grad[c];
    float d_mu = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      d_mu += out_delta[i] * -inv_sqrt_sigma + d_sigma * -2.0f * inv_var_norm * (in_act[i] - mu);
    }
    d_mu_cache[bank_idx] = d_mu;
  } else {
    d_mu_cache[bank_idx] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      d_mu_cache[bank_idx] += d_mu_cache[bank_idx+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      d_mu_cache[bank_idx] += d_mu_cache[bank_idx+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      d_mu_cache[bank_idx] += d_mu_cache[bank_idx+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      d_mu_cache[bank_idx] += d_mu_cache[bank_idx+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float d_mu = d_mu_cache[bank_idx] + d_mu_cache[bank_idx+16];
      atomicAdd(&mean_grad[c], d_mu);
    }
  }
}

__global__ void conv_bnorm_bwd_data_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *mean,
    const float *mean_grad,
    const float *var,
    const float *var_grad,
    float epsilon,
    float *in_delta)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float inv_mean_norm = 1.0f / ((float)(spatial_dim) * (float)(batch_size));
    float inv_var_norm = 1.0f / ((float)(spatial_dim - 1) * (float)(batch_size - 1));
    float mu = mean[c];
    float d_mu = mean_grad[c];
    float sigma = var[c];
    float d_sigma = var_grad[c];
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      in_delta[i] = out_delta[i] * rsqrtf(sigma + epsilon) + d_mu * inv_mean_norm + d_sigma * 2.0f * inv_var_norm * (in_act[i] - mu);
    }
  }
}

extern "C" void neuralops_cuda_conv2d_batchnorm_bwd(
    const float *in_act,
    size_t spatial_dim,
    size_t num_channels,
    size_t batch_size,
    const float *out_delta,
    const float *mean,
    const float *var,
    float epsilon,
    float *mean_grad,
    float *var_grad,
    float *in_delta,
    cudaStream_t stream)
{
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int n = 32 * num_channels * block_spatial_dim * batch_size;
  conv_bnorm_bwd_var_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, mean, var, epsilon, var_grad);
  conv_bnorm_bwd_mean_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, mean, var, var_grad, epsilon, mean_grad);
  conv_bnorm_bwd_data_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, mean, mean_grad, var, var_grad, epsilon, in_delta);
}
