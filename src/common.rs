use devicemem_cuda::prelude::*;
use operator::{OpCapability};

use std::cell::{Cell, RefCell};
use std::rc::{Rc};

pub trait DeviceOperator {
  fn _output(&self, arm: usize) -> DeviceOutput;
  fn _dev_load_diff_param<'a>(&mut self, init_offset: usize, param_reader: &mut DeviceMemRefMut<'a, f32>) -> usize { 0 }
  fn _dev_store_diff_param<'a>(&mut self, init_offset: usize, param_writer: &mut DeviceMemRefMut<'a, f32>) -> usize { 0 }
  fn _dev_store_grad<'a>(&mut self, init_offset: usize, grad_writer: &mut DeviceMemRefMut<'a, f32>) -> usize { 0 }
}

#[derive(Clone)]
pub struct DeviceOutput {
  pub batch_sz: Rc<Cell<usize>>,
  pub buf:      Rc<RefCell<DeviceMem<f32>>>,
  pub grad:     Option<Rc<RefCell<DeviceMem<f32>>>>,
}

impl DeviceOutput {
  pub fn new(batch_size: usize, frame_size: usize, cap: OpCapability, conn: DeviceConn) -> Self {
    let out_len = batch_size * frame_size;
    let out_buf = Rc::new(RefCell::new(DeviceMem::zeros(out_len, conn.clone())));
    let out_grad = if cap.enable_backward() {
      Some(Rc::new(RefCell::new(DeviceMem::zeros(out_len, conn))))
    } else {
      None
    };
    DeviceOutput{
      batch_sz: Rc::new(Cell::new(batch_size)),
      buf:      out_buf,
      grad:     out_grad,
    }
  }
}
