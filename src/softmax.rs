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

  pub fn _forward<'a>(&self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, labels: DeviceMemRef<'a, u32>, weights: DeviceMemRef<'a, f32>, targets: DeviceMemRef<'a, f32>, mut out_buf: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    assert!(batch_size <= self.batch_sz);

    in_buf.wait(&conn);
    labels.wait(&conn);
    weights.wait(&conn);
    targets.wait(&conn);
    out_buf.wait(&conn);

    // FIXME

    unsafe { neuralops_cuda_softmax_nll_loss_fwd(
        in_buf.as_ptr(),
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
