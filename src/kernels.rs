use cuda::ffi::runtime::{cudaStream_t};
use libc::*;

#[link(name = "neuralops_cuda_kernels", kind = "static")]
extern "C" {
  pub fn neuralops_cuda_image2d_crop(
      in_pixels: *const f32,
      in_width: size_t, in_height: size_t, channels: size_t,
      x_offset: ptrdiff_t, y_offset: ptrdiff_t,
      out_pixels: *mut f32,
      crop_width: size_t, crop_height: size_t,
      stream: cudaStream_t);
  pub fn neuralops_cuda_interpolate2d_catmullrom(
      in_pixels: *const f32,
      in_width: size_t, in_height: size_t, channels: size_t,
      out_pixels: *mut f32,
      out_width: size_t, out_height: size_t,
      stream: cudaStream_t);
}
