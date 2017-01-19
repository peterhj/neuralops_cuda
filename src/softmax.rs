//use prelude::*;
use kernels::*;

use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use neuralops::prelude::*;

use std::ptr::{null_mut};

pub struct DeviceSoftmaxNLLKernel {
  batch_sz:     usize,
  in_dim:       usize,
  logit:        DeviceMem<f32>,
  max_logit:    DeviceMem<f32>,
  factor:       DeviceMem<f32>,
  r_factor:     DeviceMem<f32>,
  sum_factor:   DeviceMem<f32>,
  sum_r_factor: DeviceMem<f32>,
  r_mean:       DeviceMem<f32>,
}

impl DeviceSoftmaxNLLKernel {
  pub fn new(batch_sz: usize, in_dim: usize, conn: DeviceConn) -> Self {
    DeviceSoftmaxNLLKernel{
      batch_sz:     batch_sz,
      in_dim:       in_dim,
      logit:        DeviceMem::zeros(batch_sz * in_dim, conn.clone()),
      max_logit:    DeviceMem::zeros(batch_sz, conn.clone()),
      factor:       DeviceMem::zeros(batch_sz * in_dim, conn.clone()),
      r_factor:     DeviceMem::zeros(batch_sz * in_dim, conn.clone()),
      sum_factor:   DeviceMem::zeros(batch_sz, conn.clone()),
      sum_r_factor: DeviceMem::zeros(batch_sz, conn.clone()),
      r_mean:       DeviceMem::zeros(batch_sz, conn.clone()),
    }
  }

  pub fn _forward<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, labels: DeviceMemRef<'a, u32>, weights: DeviceMemRef<'a, f32>, mut hats: DeviceMemRefMut<'a, u32>, mut probs: DeviceMemRefMut<'a, f32>, mut loss: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    assert!(batch_size <= self.batch_sz);
    assert!(self.in_dim <= 1024);

    // FIXME: copy only `batch_size * self.in_dim` amount.
    self.logit.as_mut().slice_mut(0, batch_size * self.in_dim).copy(in_buf.clone().slice(0, batch_size * self.in_dim), conn.clone());

    in_buf.wait(&conn);
    hats.wait(&conn);
    self.logit.as_ref().wait(&conn);
    self.max_logit.as_ref().wait(&conn);

    unsafe { neuralops_cuda_blockreduce_max_argmax(
        self.logit.as_ref().as_ptr(),
        self.in_dim,
        batch_size,
        self.max_logit.as_mut().as_mut_ptr(),
        hats.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    unsafe { neuralops_cuda_batchmap_add(
        self.logit.as_mut().as_mut_ptr(),
        self.in_dim,
        batch_size,
        -1.0,
        self.max_logit.as_ref().as_ptr(),
        conn.raw_stream().ptr,
    ) };

    in_buf.post(&conn);
    hats.post(&conn);
    self.logit.as_ref().post(&conn);
    self.max_logit.as_ref().post(&conn);

    // FIXME: copy only `batch_size * self.in_dim` amount.
    self.factor.as_mut().copy(self.logit.as_ref(), conn.clone());
    self.factor.as_mut().reshape_mut(batch_size * self.in_dim).exp(conn.clone());

    labels.wait(&conn);
    weights.wait(&conn);
    loss.wait(&conn);
    self.factor.as_ref().wait(&conn);
    self.sum_factor.as_ref().wait(&conn);

    unsafe { neuralops_cuda_blockreduce_sum(
        self.factor.as_ref().as_ptr(),
        self.in_dim,
        batch_size,
        0.0,
        self.sum_factor.as_mut().as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    unsafe { neuralops_cuda_batchmap_div(
        self.factor.as_mut().as_mut_ptr(),
        self.in_dim,
        batch_size,
        self.sum_factor.as_ref().as_ptr(),
        conn.raw_stream().ptr,
    ) };
    unsafe { neuralops_cuda_softmax_nll_loss_fwd(
        self.factor.as_ref().as_ptr(),
        self.in_dim,
        batch_size,
        labels.as_ptr(),
        weights.as_ptr(),
        loss.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };

    labels.post(&conn);
    weights.post(&conn);
    loss.post(&conn);
    self.factor.as_ref().post(&conn);
    self.sum_factor.as_ref().post(&conn);

    probs.slice_mut(0, batch_size * self.in_dim).copy(self.factor.as_ref().slice(0, batch_size * self.in_dim), conn.clone());
  }

  pub fn _backward<'a>(&'a self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, labels: DeviceMemRef<'a, u32>, weights: DeviceMemRef<'a, f32>, targets: DeviceMemRef<'a, f32>, mut in_grad: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    assert!(batch_size <= self.batch_sz);

    in_buf.wait(&conn);
    labels.wait(&conn);
    weights.wait(&conn);
    targets.wait(&conn);
    in_grad.wait(&conn);

    unsafe { neuralops_cuda_softmax_nll_loss_bwd(
        in_buf.as_ptr(),
        self.in_dim,
        batch_size,
        labels.as_ptr(),
        weights.as_ptr(),
        targets.as_ptr(),
        in_grad.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };

    in_buf.post(&conn);
    labels.post(&conn);
    weights.post(&conn);
    targets.post(&conn);
    in_grad.post(&conn);
  }

  pub fn _backward2<'a>(&'a self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, labels: DeviceMemRef<'a, u32>, weights: DeviceMemRef<'a, f32>, targets: DeviceMemRef<'a, f32>, mut in_grad2: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    assert!(batch_size <= self.batch_sz);

    in_buf.wait(&conn);
    labels.wait(&conn);
    weights.wait(&conn);
    targets.wait(&conn);
    in_grad2.wait(&conn);

    unsafe { neuralops_cuda_softmax_nll_loss_bwd2(
        in_buf.as_ptr(),
        self.in_dim,
        batch_size,
        labels.as_ptr(),
        weights.as_ptr(),
        targets.as_ptr(),
        in_grad2.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };

    in_buf.post(&conn);
    labels.post(&conn);
    weights.post(&conn);
    targets.post(&conn);
    in_grad2.post(&conn);
  }

  pub fn _r_forward<'a>(&'a mut self, batch_size: usize, in_val: DeviceMemRef<'a, f32>, in_r_val: DeviceMemRef<'a, f32>, labels: DeviceMemRef<'a, u32>, mut r_loss: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    assert!(batch_size <= self.batch_sz);
    assert!(self.in_dim <= 1024);

    // FIXME: copy only `batch_size * self.in_dim` amount.
    self.logit.as_mut().slice_mut(0, batch_size * self.in_dim).copy(in_val.clone().slice(0, batch_size * self.in_dim), conn.clone());

    //hats.wait(&conn);
    self.logit.as_ref().wait(&conn);
    self.max_logit.as_ref().wait(&conn);

    unsafe { neuralops_cuda_blockreduce_max_argmax(
        self.logit.as_ref().as_ptr(),
        self.in_dim,
        batch_size,
        self.max_logit.as_mut().as_mut_ptr(),
        //hats.as_mut_ptr(),
        null_mut(),
        conn.raw_stream().ptr,
    ) };
    unsafe { neuralops_cuda_batchmap_add(
        self.logit.as_mut().as_mut_ptr(),
        self.in_dim,
        batch_size,
        -1.0,
        self.max_logit.as_ref().as_ptr(),
        conn.raw_stream().ptr,
    ) };

    //hats.post(&conn);
    self.logit.as_ref().post(&conn);
    self.max_logit.as_ref().post(&conn);

    // FIXME: copy only `batch_size * self.in_dim` amount.
    self.factor.as_mut().copy(self.logit.as_ref(), conn.clone());
    self.factor.as_mut().reshape_mut(batch_size * self.in_dim).exp(conn.clone());
    self.r_factor.as_mut().copy(self.factor.as_ref(), conn.clone());
    self.r_factor.as_mut().reshape_mut(batch_size * self.in_dim).elem_mult(in_r_val.clone().reshape(batch_size * self.in_dim), conn.clone());

    self.factor.as_ref().wait(&conn);
    self.r_factor.as_ref().wait(&conn);
    self.sum_factor.as_ref().wait(&conn);
    self.sum_r_factor.as_ref().wait(&conn);

    unsafe { neuralops_cuda_blockreduce_sum(
        self.factor.as_ref().as_ptr(),
        self.in_dim,
        batch_size,
        0.0,
        self.sum_factor.as_mut().as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    unsafe { neuralops_cuda_blockreduce_sum(
        self.r_factor.as_ref().as_ptr(),
        self.in_dim,
        batch_size,
        0.0,
        self.sum_r_factor.as_mut().as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };

    self.factor.as_ref().post(&conn);
    self.r_factor.as_ref().post(&conn);
    self.sum_factor.as_ref().post(&conn);
    self.sum_r_factor.as_ref().post(&conn);

    self.r_mean.as_mut().copy(self.sum_r_factor.as_ref(), conn.clone());
    self.r_mean.as_mut().reshape_mut(batch_size).elem_div(self.sum_factor.as_ref().reshape(batch_size), conn.clone());

    in_r_val.wait(&conn);
    labels.wait(&conn);
    r_loss.wait(&conn);
    self.r_mean.as_ref().wait(&conn);

    unsafe { neuralops_cuda_softmax_nll_loss_rfwd(
        in_r_val.as_ptr(),
        self.in_dim,
        batch_size,
        self.r_mean.as_ref().as_ptr(),
        labels.as_ptr(),
        r_loss.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };

    in_r_val.post(&conn);
    labels.post(&conn);
    r_loss.post(&conn);
    self.r_mean.as_ref().post(&conn);
  }
}
