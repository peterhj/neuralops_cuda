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
  pub fn neuralops_cuda_activate_rect_bwd2(
      in_act: *const f32,
      dim: size_t,
      out_delta: *const f32,
      in_delta: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_activate_rect_rfwd(
      in_val: *const f32,
      dim: size_t,
      in_r_val: *const f32,
      out_r_val: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_activate_rect_rbwd(
      in_val: *const f32,
      dim: size_t,
      out_r_grad: *const f32,
      in_r_grad: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_activate_leakrect_fwd(
      in_act: *const f32,
      dim: size_t,
      out_act: *mut f32,
      neg_slope: f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_activate_leakrect_bwd(
      in_act: *const f32,
      dim: size_t,
      out_delta: *const f32,
      in_delta: *mut f32,
      neg_slope: f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_activate_logistic_fwd(
      in_act: *const f32,
      dim: size_t,
      out_act: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_activate_logistic_bwd(
      in_act: *const f32,
      dim: size_t,
      out_delta: *const f32,
      in_delta: *mut f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_cast_u8_to_f32(
      x: *const u8,
      dim: size_t,
      y: *mut f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_clamp(
      y: *mut f32,
      dim: size_t,
      clamp_lo: f32,
      clamp_hi: f32,
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
  pub fn neuralops_cuda_conv_scale_rfwd(
      in_val: *const f32,
      spatial_dim: size_t,
      num_channels: size_t,
      batch_size: size_t,
      in_r_val: *const f32,
      scale: *const f32,
      scale_r_dir: *const f32,
      bias_r_dir: *const f32,
      out_r_val: *mut f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_caffe_im2col(
      data_im: *const f32,
      channels: c_int,
      height: c_int,
      width: c_int,
      kernel_h: c_int,
      kernel_w: c_int,
      pad_h: c_int,
      pad_w: c_int,
      stride_h: c_int,
      stride_w: c_int,
      dilation_h: c_int,
      dilation_w: c_int,
      data_col: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_caffe_col2im(
      data_col: *const f32,
      channels: c_int,
      height: c_int,
      width: c_int,
      kernel_h: c_int,
      kernel_w: c_int,
      pad_h: c_int,
      pad_w: c_int,
      stride_h: c_int,
      stride_w: c_int,
      dilation_h: c_int,
      dilation_w: c_int,
      data_im: *mut f32,
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

  pub fn neuralops_cuda_gaussian_kl_loss_fwd(
      mean: *const f32,
      batch_sz: u32,
      target_mean: *const f32,
      var: f32,
      target_var: f32,
      loss: *mut f32,
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

  pub fn neuralops_cuda_linear_bias_fwd(
      in_buf: *const f32,
      dim: size_t,
      batch_size: size_t,
      bias: *const f32,
      out_buf: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_linear_bias_fwd_inplace(
      out_buf: *mut f32,
      dim: size_t,
      batch_size: size_t,
      bias: *const f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_linear_bias_bwd(
      out_grad: *const f32,
      dim: size_t,
      batch_size: size_t,
      in_grad: *mut f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_logistic_nll_loss_fwd(
      in_buf: *const f32,
      batch_size: size_t,
      labels: *const u32,
      weights: *const f32,
      losses: *mut f32,
      preds: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_logistic_nll_loss_bwd(
      in_buf: *const f32,
      batch_size: size_t,
      labels: *const u32,
      weights: *const f32,
      in_delta: *mut f32,
      stream: cudaStream_t);

  pub fn neuralops_cuda_lst_sq_fwd(
      in_buf: *const f32,
      batch_sz: size_t,
      targets: *const f32,
      weights: *const f32,
      loss: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_lst_sq_bwd(
      in_buf: *const f32,
      batch_sz: size_t,
      targets: *const f32,
      weights: *const f32,
      in_grad: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_lst_sq_rfwd(
      in_buf: *const f32,
      batch_sz: size_t,
      r_in_buf: *const f32,
      targets: *const f32,
      jac_targ: *const f32,
      r_loss: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_ind_lst_sq_fwd(
      in_buf: *const f32,
      dim: size_t,
      batch_sz: size_t,
      targets: *const f32,
      labels: *const u32,
      weights: *const f32,
      loss: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_ind_lst_sq_bwd(
      in_buf: *const f32,
      dim: size_t,
      batch_sz: size_t,
      targets: *const f32,
      labels: *const u32,
      weights: *const f32,
      in_grad: *mut f32,
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

  pub fn neuralops_cuda_caffe_avgpool2d_fwd(
      bottom_data: *const f32,
      num: c_int, channels: c_int, height: c_int, width: c_int,
      pool_h: c_int, pool_w: c_int,
      kernel_h: c_int, kernel_w: c_int,
      pad_h: c_int, pad_w: c_int,
      stride_h: c_int, stride_w: c_int,
      top_data: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_caffe_avgpool2d_bwd(
      top_diff: *const f32,
      num: c_int, channels: c_int, height: c_int, width: c_int,
      pool_h: c_int, pool_w: c_int,
      kernel_h: c_int, kernel_w: c_int,
      pad_h: c_int, pad_w: c_int,
      stride_h: c_int, stride_w: c_int,
      bottom_diff: *mut f32,
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

  pub fn neuralops_cuda_softmax_lr_loss_fwd(
      ys: *const f32,
      dim: u32,
      batch_sz: u32,
      labels: *const u32,
      targets: *const f32,
      weights: *const f32,
      cutoff: f32,
      loss: *mut f32,
      stream: cudaStream_t,
  );
  pub fn neuralops_cuda_softmax_lr_loss_bwd(
      ys: *const f32,
      dim: u32,
      batch_sz: u32,
      labels: *const u32,
      targets: *const f32,
      weights: *const f32,
      cutoff: f32,
      grad: *mut f32,
      stream: cudaStream_t,
  );
  pub fn neuralops_cuda_softmax_kl_loss_fwd(
      ys: *const f32,
      dim: u32,
      batch_sz: u32,
      targets: *const f32,
      epsilon: f32,
      loss: *mut f32,
      stream: cudaStream_t,
  );
  pub fn neuralops_cuda_softmax_kl_loss_bwd(
      ys: *const f32,
      dim: u32,
      batch_sz: u32,
      targets: *const f32,
      grad: *mut f32,
      stream: cudaStream_t,
  );
  pub fn neuralops_cuda_softmax_kl_loss_rfwd(
      ys: *const f32,
      dim: u32,
      batch_sz: u32,
      r_xs: *const f32,
      r_mean: *const f32,
      targets: *const f32,
      r_loss: *mut f32,
      r_grad: *mut f32,
      stream: cudaStream_t,
  );
  pub fn neuralops_cuda_softmax_nll_loss_fwd(
      in_buf: *const f32,
      num_classes: size_t,
      batch_size: size_t,
      labels: *const u32,
      weights: *const f32,
      out_buf: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_softmax_nll_loss_bwd(
      out_buf: *const f32,
      num_classes: size_t,
      batch_size: size_t,
      labels: *const u32,
      weights: *const f32,
      jac_targ: *const f32,
      in_delta: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_softmax_nll_loss_bwd2(
      out_buf: *const f32,
      num_classes: size_t,
      batch_size: size_t,
      labels: *const u32,
      weights: *const f32,
      jac_targ: *const f32,
      in_delta2: *mut f32,
      stream: cudaStream_t);
  pub fn neuralops_cuda_softmax_nll_loss_rfwd(
      out_r_val: *const f32,
      num_classes: size_t,
      batch_size: size_t,
      r_mean: *const f32,
      labels: *const u32,
      r_loss: *mut f32,
      stream: cudaStream_t);
}
