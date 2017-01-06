use devicemem_cuda::prelude::*;
use operator::{OpCapability};

use std::cell::{Cell, RefCell, RefMut};
use std::rc::{Rc};

pub trait DeviceOperator {
  fn _output(&self, arm: usize) -> DeviceOutput;
  fn _dev_load_diff_param<'a>(&mut self, init_offset: usize, param_reader: &mut DeviceMemRefMut<'a, f32>) -> usize { 0 }
  fn _dev_store_diff_param<'a>(&mut self, init_offset: usize, param_writer: &mut DeviceMemRefMut<'a, f32>) -> usize { 0 }
  fn _dev_store_grad<'a>(&mut self, init_offset: usize, grad_writer: &mut DeviceMemRefMut<'a, f32>) -> usize { 0 }
}

#[derive(Clone)]
pub struct DeviceOutput {
  data_dim:     usize,
  pub batch_sz: Rc<Cell<usize>>,
  pub buf:      Rc<RefCell<DeviceMem<f32>>>,
  pub grad:     Option<Rc<RefCell<DeviceMem<f32>>>>,
  pub grad2:    Rc<RefCell<Option<DeviceMem<f32>>>>,
  pub r_data:   Rc<RefCell<Option<DeviceMem<f32>>>>,
  pub r_grad:   Rc<RefCell<Option<DeviceMem<f32>>>>,
}

impl DeviceOutput {
  pub fn new(batch_size: usize, data_dim: usize, cap: OpCapability, conn: DeviceConn) -> Self {
    let out_len = batch_size * data_dim;
    let out_buf = Rc::new(RefCell::new(DeviceMem::zeros(out_len, conn.clone())));
    let out_grad = if cap.enable_backward() {
      Some(Rc::new(RefCell::new(DeviceMem::zeros(out_len, conn.clone()))))
    } else {
      None
    };
    /*let r_data = if cap.enable_r_forward() {
      Some(Rc::new(RefCell::new(DeviceMem::zeros(out_len, conn.clone()))))
    } else {
      None
    };*/
    DeviceOutput{
      data_dim: data_dim,
      batch_sz: Rc::new(Cell::new(batch_size)),
      buf:      out_buf,
      grad:     out_grad,
      grad2:    Rc::new(RefCell::new(None)),
      r_data:   Rc::new(RefCell::new(None)),
      r_grad:   Rc::new(RefCell::new(None)),
    }
  }

  pub fn data(&self, conn: DeviceConn) -> RefMut<DeviceMem<f32>> {
    unimplemented!();
  }

  pub fn grad(&self, conn: DeviceConn) -> RefMut<DeviceMem<f32>> {
    unimplemented!();
  }

  pub fn grad2(&self, conn: DeviceConn) -> RefMut<DeviceMem<f32>> {
    {
      let mut grad2 = self.grad2.borrow_mut();
      if grad2.is_none() {
        *grad2 = Some(DeviceMem::zeros(self.data_dim * self.batch_sz.get(), conn));
      }
    }
    RefMut::map(self.grad2.borrow_mut(), |x| x.as_mut().unwrap())
  }

  pub fn r_data(&self, conn: DeviceConn) -> RefMut<DeviceMem<f32>> {
    {
      let mut r_data = self.r_data.borrow_mut();
      if r_data.is_none() {
        *r_data = Some(DeviceMem::zeros(self.data_dim * self.batch_sz.get(), conn));
      }
    }
    RefMut::map(self.r_data.borrow_mut(), |x| x.as_mut().unwrap())
  }

  pub fn r_grad(&self, conn: DeviceConn) -> RefMut<DeviceMem<f32>> {
    {
      let mut r_grad = self.r_grad.borrow_mut();
      if r_grad.is_none() {
        *r_grad = Some(DeviceMem::zeros(self.data_dim * self.batch_sz.get(), conn));
      }
    }
    RefMut::map(self.r_grad.borrow_mut(), |x| x.as_mut().unwrap())
  }
}
