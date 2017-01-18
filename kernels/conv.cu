#include <cuda_runtime_api.h>
#include <stdint.h>

#define OFFSET_BANK(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) / 32)); })

__global__ void im2col_gpu_kernel(
    const int n,
    const float* data_im,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    float* data_col)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

extern "C" void neuralops_cuda_caffe_im2col(
    const float* data_im,
    const int channels,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col,
    cudaStream_t stream)
{
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<<<(num_kernels+1024-1)/1024, 1024, 0, stream>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
}

__global__ void col2im_gpu_kernel(
    const int n,
    const float* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    float* data_im)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < n) {
    float val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

extern "C" void neuralops_cuda_caffe_col2im(
    const float* data_col,
    const int channels,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im,
    cudaStream_t stream)
{
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<<<(num_kernels+1024-1)/1024, 1024, 0, stream>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im);
}

__global__ void conv_diag_affine_fwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *scale,
    const float *bias,
    float *out_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int u = idx % spatial_dim;
  int c = (idx / spatial_dim) % num_channels;
  int batch_idx = idx / (spatial_dim * num_channels);
  if (u < spatial_dim && c < num_channels && batch_idx < batch_size) {
    float gamma = scale[c];
    float beta = bias[c];
    float y = gamma * in_act[idx] + beta;
    out_act[idx] = y;
  }
}

extern "C" void neuralops_cuda_conv2d_scale_fwd(
    const float *in_act,
    size_t spatial_dim,
    size_t num_channels,
    size_t batch_size,
    const float *scale,
    const float *bias,
    float *out_act,
    cudaStream_t stream)
{
  int n = spatial_dim * num_channels * batch_size;
  conv_diag_affine_fwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, scale, bias, out_act);
}

__global__ void conv_diag_affine_bwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *scale,
    float *scale_grad,
    float *bias_grad,
    float *in_delta)
{
  __shared__ float scale_grad_cache[1024+32];
  __shared__ float bias_grad_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float gamma = scale[c];
    float d_gamma = 0.0f;
    float d_beta = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float dy = out_delta[i];
      d_gamma += dy * in_act[i];
      d_beta += dy;
      in_delta[i] = dy * gamma;
      //in_delta[i] += dy * gamma;
      //atomicAdd(&in_delta[i], dy * gamma);
    }
    scale_grad_cache[bank_idx] = d_gamma;
    bias_grad_cache[bank_idx] = d_beta;
  } else {
    scale_grad_cache[bank_idx] = 0.0f;
    bias_grad_cache[bank_idx] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+1];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+2];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+4];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+8];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float d_gamma = scale_grad_cache[bank_idx] + scale_grad_cache[bank_idx+16];
      atomicAdd(&scale_grad[c], d_gamma);
      float d_beta = bias_grad_cache[bank_idx] + bias_grad_cache[bank_idx+16];
      atomicAdd(&bias_grad[c], d_beta);
    }
  }
}

extern "C" void neuralops_cuda_conv2d_scale_bwd(
    const float *in_act,
    size_t spatial_dim,
    size_t num_channels,
    size_t batch_size,
    const float *out_delta,
    const float *scale,
    float *scale_grad,
    float *bias_grad,
    float *in_delta,
    cudaStream_t stream)
{
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int n = 32 * num_channels * block_spatial_dim * batch_size;
  conv_diag_affine_bwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, scale, scale_grad, bias_grad, in_delta);
}

__global__ void conv_scale_rfwd_kernel(
    const float *in_val,
    uint32_t spatial_dim,
    uint32_t num_channels,
    uint32_t batch_size,
    const float *in_r_val,
    const float *scale,
    const float *scale_r_dir,
    const float *bias_r_dir,
    float *out_r_val)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t u = idx % spatial_dim;
  uint32_t c = (idx / spatial_dim) % num_channels;
  uint32_t batch_idx = idx / (spatial_dim * num_channels);
  if (u < spatial_dim && c < num_channels && batch_idx < batch_size) {
    float alpha = scale[c];
    float r_alpha = scale_r_dir[c];
    float r_beta = bias_r_dir[c];
    float r_y = alpha * in_r_val[idx] + r_alpha * in_val[idx] + r_beta;
    out_r_val[idx] = r_y;
  }
}

extern "C" void neuralops_cuda_conv_scale_rfwd(
    const float *in_val,
    size_t spatial_dim,
    size_t num_channels,
    size_t batch_size,
    const float *in_r_val,
    const float *scale,
    const float *scale_r_dir,
    const float *bias_r_dir,
    float *out_r_val,
    cudaStream_t stream)
{
  uint32_t n = spatial_dim * num_channels * batch_size;
  conv_scale_rfwd_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_val, spatial_dim, num_channels, batch_size, in_r_val, scale, scale_r_dir, bias_r_dir, out_r_val);
}
