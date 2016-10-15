extern crate cuda;
extern crate libc;

use cuda::ffi::runtime::{cudaStream_t};
use libc::*;

#[link(name = "neuralops_cuda_kernels", kind = "static")]
extern "C" {
  pub fn neuralops_cuda_activate_rect_fwd(
      in_act: *const f32,
      dim: size_t,
      out_act: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_activate_rect_bwd(
      in_act: *const f32,
      dim: size_t,
      out_delta: *const f32,
      in_delta: *mut f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_conv2d_scale_fwd(
      in_buf: *const f32,
      spatial_dim: size_t,
      num_channels: size_t,
      batch_size: size_t,
      scale: *const f32,
      bias: *const f32,
      out_buf: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_conv2d_scale_bwd(
      in_buf: *const f32,
      spatial_dim: size_t,
      num_channels: size_t,
      batch_size: size_t,
      out_grad: *const f32,
      scale: *const f32,
      scale_grad: *mut f32,
      bias_grad: *mut f32,
      in_grad: *mut f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_conv2d_whiten_fwd(
      in_buf: *const f32,
      spatial_dim: size_t,
      num_channels: size_t,
      batch_size: size_t,
      mean: *const f32,
      var: *const f32,
      epsilon: f32,
      out_buf: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_conv2d_mean_fwd(
      in_buf: *const f32,
      spatial_dim: size_t,
      num_channels: size_t,
      batch_size: size_t,
      mean: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_conv2d_var_fwd(
      in_buf: *const f32,
      spatial_dim: size_t,
      num_channels: size_t,
      batch_size: size_t,
      mean: *const f32,
      var: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_conv2d_batchnorm_bwd(
      in_buf: *const f32,
      spatial_dim: size_t,
      num_channels: size_t,
      batch_size: size_t,
      out_grad: *const f32,
      mean: *const f32,
      var: *const f32,
      epsilon: f32,
      mean_grad: *mut f32,
      var_grad: *mut f32,
      in_grad: *mut f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_image2d_crop(
      in_pixels: *const f32,
      in_width: size_t, in_height: size_t, channels: size_t,
      x_offset: ptrdiff_t, y_offset: ptrdiff_t,
      out_pixels: *mut f32,
      crop_width: size_t, crop_height: size_t,
      stream: cudaStream_t);
  pub fn neuralops_cuda_image2d_flip(
      in_pixels: *const f32,
      in_width: size_t, in_height: size_t, channels: size_t,
      out_pixels: *mut f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_interpolate2d_catmullrom(
      in_pixels: *const f32,
      in_width: size_t, in_height: size_t, channels: size_t,
      out_pixels: *mut f32,
      out_width: size_t, out_height: size_t,
      stream: cudaStream_t);

  pub fn neuralops_cuda_batchmap_add(
      xs: *mut f32,
      len: size_t,
      batch_size: size_t,
      alpha: f32,
      scalars: *const f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_batchmap_div(
      xs: *mut f32,
      len: size_t,
      batch_size: size_t,
      scalars: *const f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_blockreduce_max_argmax(
      xs: *const f32,
      len: size_t,
      batch_size: size_t,
      xs_max: *mut f32,
      xs_argmax: *mut u32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_blockreduce_sum(
      xs: *const f32,
      len: size_t,
      batch_size: size_t,
      alpha: f32,
      xs_sum: *mut f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_softmax_nll_loss_fwd(
      in_buf: *const f32,
      num_classes: size_t,
      batch_size: size_t,
      labels: *const u32,
      weights: *const f32,
      targets: *const f32,
      out_buf: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_softmax_nll_loss_bwd(
      in_buf: *const f32,
      num_classes: size_t,
      batch_size: size_t,
      labels: *const u32,
      weights: *const f32,
      targets: *const f32,
      in_delta: *mut f32,
      stream: cudaStream_t);
}
