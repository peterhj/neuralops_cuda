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
