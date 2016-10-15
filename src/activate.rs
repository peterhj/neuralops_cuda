//use prelude::*;
use kernels::*;

//use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use neuralops::prelude::*;

pub struct DeviceActivateKernel {
  pub out_dim:  usize,
  pub act_kind: ActivationKind,
}

impl DeviceActivateKernel {
  pub fn new(out_dim: usize, act_kind: ActivationKind) -> Self {
    DeviceActivateKernel{
      out_dim:  out_dim,
      act_kind: act_kind,
    }
  }

  pub fn _forward<'a>(&self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, mut out_buf: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    in_buf.wait(&conn);
    out_buf.wait(&conn);
    match self.act_kind {
      ActivationKind::Identity => {
        out_buf.copy(in_buf.clone(), conn.clone());
      }
      ActivationKind::Rect => {
        unsafe { neuralops_cuda_activate_rect_fwd(
            in_buf.as_ptr(),
            batch_size * self.out_dim,
            out_buf.as_mut_ptr(),
            conn.stream().ptr,
        ) };
      }
      _ => unimplemented!(),
    }
    in_buf.post(&conn);
    out_buf.post(&conn);
  }

  pub fn _backward<'a>(&self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, out_grad: DeviceMemRef<'a, f32>, mut in_grad: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    in_buf.wait(&conn);
    out_grad.wait(&conn);
    in_grad.wait(&conn);
    match self.act_kind {
      ActivationKind::Identity => {
        in_grad.copy(out_grad.clone(), conn.clone());
      }
      ActivationKind::Rect => {
        unsafe { neuralops_cuda_activate_rect_bwd(
            in_buf.as_ptr(),
            batch_size * self.out_dim,
            out_grad.as_ptr(),
            in_grad.as_mut_ptr(),
            conn.stream().ptr,
        ) };
      }
      _ => unimplemented!(),
    }
    in_buf.post(&conn);
    out_grad.post(&conn);
    in_grad.post(&conn);
  }
}
