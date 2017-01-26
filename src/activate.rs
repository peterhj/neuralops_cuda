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
    match self.act_kind {
      ActivationKind::Identity => {
        out_buf.slice_mut(0, batch_size * self.out_dim)
          .copy(in_buf.clone().slice(0, batch_size * self.out_dim), conn.clone());
      }
      ActivationKind::Rect => {
        in_buf.wait(&conn);
        out_buf.wait(&conn);
        unsafe { neuralops_cuda_activate_rect_fwd(
            in_buf.as_ptr(),
            batch_size * self.out_dim,
            out_buf.as_mut_ptr(),
            conn.raw_stream().ptr,
        ) };
        in_buf.post(&conn);
        out_buf.post(&conn);
      }
      ActivationKind::LeakyRect(neg_slope) => {
        in_buf.wait(&conn);
        out_buf.wait(&conn);
        unsafe { neuralops_cuda_activate_leakrect_fwd(
            in_buf.as_ptr(),
            batch_size * self.out_dim,
            out_buf.as_mut_ptr(),
            neg_slope,
            conn.raw_stream().ptr,
        ) };
        in_buf.post(&conn);
        out_buf.post(&conn);
      }
      _ => unimplemented!(),
    }
  }

  pub fn _backward<'a>(&self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, out_grad: DeviceMemRef<'a, f32>, mut in_grad: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    match self.act_kind {
      ActivationKind::Identity => {
        in_grad.slice_mut(0, batch_size * self.out_dim)
          .copy(out_grad.clone().slice(0, batch_size * self.out_dim), conn.clone());
      }
      ActivationKind::Rect => {
        in_buf.wait(&conn);
        out_grad.wait(&conn);
        in_grad.wait(&conn);
        unsafe { neuralops_cuda_activate_rect_bwd(
            in_buf.as_ptr(),
            batch_size * self.out_dim,
            out_grad.as_ptr(),
            in_grad.as_mut_ptr(),
            conn.raw_stream().ptr,
        ) };
        in_buf.post(&conn);
        out_grad.post(&conn);
        in_grad.post(&conn);
      }
      ActivationKind::LeakyRect(neg_slope) => {
        in_buf.wait(&conn);
        out_grad.wait(&conn);
        in_grad.wait(&conn);
        unsafe { neuralops_cuda_activate_leakrect_bwd(
            in_buf.as_ptr(),
            batch_size * self.out_dim,
            out_grad.as_ptr(),
            in_grad.as_mut_ptr(),
            neg_slope,
            conn.raw_stream().ptr,
        ) };
        in_buf.post(&conn);
        out_grad.post(&conn);
        in_grad.post(&conn);
      }
      _ => unimplemented!(),
    }
  }

  pub fn _backward2<'a>(&self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, out_grad2: DeviceMemRef<'a, f32>, mut in_grad2: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    match self.act_kind {
      ActivationKind::Identity => {
        in_grad2.slice_mut(0, batch_size * self.out_dim)
          .copy(out_grad2.clone().slice(0, batch_size * self.out_dim), conn.clone());
      }
      ActivationKind::Rect => {
        in_buf.wait(&conn);
        out_grad2.wait(&conn);
        in_grad2.wait(&conn);
        unsafe { neuralops_cuda_activate_rect_bwd2(
            in_buf.as_ptr(),
            batch_size * self.out_dim,
            out_grad2.as_ptr(),
            in_grad2.as_mut_ptr(),
            conn.raw_stream().ptr,
        ) };
        in_buf.post(&conn);
        out_grad2.post(&conn);
        in_grad2.post(&conn);
      }
      _ => unimplemented!(),
    }
  }

  pub fn _r_forward<'a>(&self, batch_size: usize, in_val: DeviceMemRef<'a, f32>, in_r_val: DeviceMemRef<'a, f32>, mut out_r_val: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    match self.act_kind {
      ActivationKind::Identity => {
        out_r_val.slice_mut(0, batch_size * self.out_dim)
          .copy(in_r_val.clone().slice(0, batch_size * self.out_dim), conn.clone());
      }
      ActivationKind::Rect => {
        in_val.wait(&conn);
        in_r_val.wait(&conn);
        out_r_val.wait(&conn);
        unsafe { neuralops_cuda_activate_rect_rfwd(
            in_val.as_ptr(),
            batch_size * self.out_dim,
            in_r_val.as_ptr(),
            out_r_val.as_mut_ptr(),
            conn.raw_stream().ptr,
        ) };
        in_val.post(&conn);
        in_r_val.post(&conn);
        out_r_val.post(&conn);
      }
      _ => unimplemented!(),
    }
  }

  pub fn _r_backward<'a>(&self, batch_size: usize, in_val: DeviceMemRef<'a, f32>, in_r_val: DeviceMemRef<'a, f32>, out_r_grad: DeviceMemRef<'a, f32>, mut in_r_grad: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    match self.act_kind {
      ActivationKind::Identity => {
        in_r_grad.slice_mut(0, batch_size * self.out_dim)
          .copy(out_r_grad.clone().slice(0, batch_size * self.out_dim), conn.clone());
      }
      ActivationKind::Rect => {
        in_val.wait(&conn);
        out_r_grad.wait(&conn);
        in_r_grad.wait(&conn);
        unsafe { neuralops_cuda_activate_rect_rbwd(
            in_val.as_ptr(),
            batch_size * self.out_dim,
            out_r_grad.as_ptr(),
            in_r_grad.as_mut_ptr(),
            conn.raw_stream().ptr,
        ) };
        in_val.post(&conn);
        out_r_grad.post(&conn);
        in_r_grad.post(&conn);
      }
      _ => unimplemented!(),
    }
  }
}
