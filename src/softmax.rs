//use prelude::*;
use kernels::*;

//use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use neuralops::prelude::*;

pub struct DeviceSoftmaxKernel {
  batch_sz:     usize,
  in_dim:       usize,
  logit:        DeviceMem<f32>,
  max_logit:    DeviceMem<f32>,
  factor:       DeviceMem<f32>,
  sum_factor:   DeviceMem<f32>,
}

impl DeviceSoftmaxKernel {
  pub fn new(batch_sz: usize, in_dim: usize, conn: DeviceConn) -> Self {
    DeviceSoftmaxKernel{
      batch_sz:     batch_sz,
      in_dim:       in_dim,
      logit:        DeviceMem::zeros(batch_sz * in_dim, conn.clone()),
      max_logit:    DeviceMem::zeros(batch_sz, conn.clone()),
      factor:       DeviceMem::zeros(batch_sz * in_dim, conn.clone()),
      sum_factor:   DeviceMem::zeros(batch_sz, conn.clone()),
    }
  }

  pub fn _forward<'a>(&mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, labels: DeviceMemRef<'a, u32>, weights: DeviceMemRef<'a, f32>, targets: DeviceMemRef<'a, f32>, mut hats: DeviceMemRefMut<'a, u32>, mut out_buf: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    assert!(batch_size <= self.batch_sz);

    in_buf.wait(&conn);
    labels.wait(&conn);
    weights.wait(&conn);
    targets.wait(&conn);
    out_buf.wait(&conn);

    // FIXME: copy only `batch_size * self.in_dim` amount.
    self.logit.as_mut().copy(in_buf.clone(), conn.clone());
    assert!(self.in_dim <= 1024);
    unsafe { neuralops_cuda_blockreduce_max_argmax(
        self.logit.as_ref().as_ptr(),
        self.in_dim,
        batch_size,
        self.max_logit.as_mut().as_mut_ptr(),
        hats.as_mut_ptr(),
        conn.stream().ptr,
    ) };
    unsafe { neuralops_cuda_batchmap_add(
        self.logit.as_mut().as_mut_ptr(),
        self.in_dim,
        batch_size,
        -1.0,
        self.max_logit.as_ref().as_ptr(),
        conn.stream().ptr,
    ) };
    // FIXME: copy only `batch_size * self.in_dim` amount.
    self.factor.as_mut().copy(self.logit.as_ref(), conn.clone());
    unsafe { neuralops_cuda_blockreduce_sum(
        self.logit.as_ref().as_ptr(),
        self.in_dim,
        batch_size,
        0.0,
        self.sum_factor.as_mut().as_mut_ptr(),
        conn.stream().ptr,
    ) };
    unsafe { neuralops_cuda_batchmap_div(
        self.factor.as_mut().as_mut_ptr(),
        self.in_dim,
        batch_size,
        self.sum_factor.as_ref().as_ptr(),
        conn.stream().ptr,
    ) };
    unsafe { neuralops_cuda_softmax_nll_loss_fwd(
        self.factor.as_ref().as_ptr(),
        self.in_dim,
        batch_size,
        labels.as_ptr(),
        weights.as_ptr(),
        targets.as_ptr(),
        out_buf.as_mut_ptr(),
        conn.stream().ptr,
    ) };

    in_buf.post(&conn);
    labels.post(&conn);
    weights.post(&conn);
    targets.post(&conn);
    out_buf.post(&conn);
  }

  pub fn _backward<'a>(&self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, labels: DeviceMemRef<'a, u32>, weights: DeviceMemRef<'a, f32>, targets: DeviceMemRef<'a, f32>, mut in_grad: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
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
        conn.stream().ptr,
    ) };

    in_buf.post(&conn);
    labels.post(&conn);
    weights.post(&conn);
    targets.post(&conn);
    in_grad.post(&conn);
  }
}
