use prelude::*;
//use activate::{DeviceActivateKernel};
use kernels::*;
//use util::*;

use cuda_dnn::v5::{CudnnPoolingOp, CudnnTensorLayout, CudnnTensorDesc};
use cuda_dnn::v5::ffi::*;
use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use operator::prelude::*;
use operator::io::{IoBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use std::cell::{RefCell};
use std::rc::{Rc};
use std::path::{PathBuf};

pub struct DeviceCaffePool2dOperator<S, IoBuf: ?Sized> {
  cfg:      Pool2dOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
}

impl<S, IoBuf: ?Sized> DeviceCaffePool2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: Pool2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceCaffePool2dOperator<S, IoBuf>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    let (in_w, in_h, chan) = cfg.in_dim;
    let (out_w, out_h, _) = cfg.out_dim();
    //println!("DEBUG: pool2d: {:?} {:?}", cfg.in_dim, cfg.out_dim());
    Rc::new(RefCell::new(DeviceCaffePool2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap, stream.clone()),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for DeviceCaffePool2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> DeviceOperator for DeviceCaffePool2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for DeviceCaffePool2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DeviceCaffePool2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for DeviceCaffePool2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);
    let in_buf = self.in_.buf.borrow();
    let mut out_buf = self.out.buf.borrow_mut();

    in_buf.as_ref().wait(&self.stream.conn());
    out_buf.as_ref().wait(&self.stream.conn());
    match self.cfg.kind {
      PoolKind::Average => {
        unsafe { neuralops_cuda_caffe_avgpool2d_fwd(
            in_buf.as_ref().as_ptr(),
            batch_size as _, self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
            self.cfg.out_dim().1 as _, self.cfg.out_dim().0 as _,
            self.cfg.pool_h as _, self.cfg.pool_w as _,
            self.cfg.pad_h as _, self.cfg.pad_w as _,
            self.cfg.stride_h as _, self.cfg.stride_w as _,
            out_buf.as_mut().as_mut_ptr(),
            self.stream.conn().raw_stream().ptr,
        ) };
      }
      _ => unimplemented!(),
    }
    in_buf.as_ref().post(&self.stream.conn());
    out_buf.as_ref().post(&self.stream.conn());
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let out_grad = self.out.grad.as_ref().unwrap().borrow();
      let mut in_grad = in_grad.borrow_mut();

      match self.cfg.kind {
        PoolKind::Average => {
          in_grad.as_mut().slice_mut(0, batch_size * self.cfg.in_dim.flat_len())
            .set_constant(0.0, self.stream.conn());
          out_grad.as_ref().wait(&self.stream.conn());
          in_grad.as_ref().wait(&self.stream.conn());
          unsafe { neuralops_cuda_caffe_avgpool2d_bwd(
              out_grad.as_ref().as_ptr(),
              batch_size as _, self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
              self.cfg.out_dim().1 as _, self.cfg.out_dim().0 as _,
              self.cfg.pool_h as _, self.cfg.pool_w as _,
              self.cfg.pad_h as _, self.cfg.pad_w as _,
              self.cfg.stride_h as _, self.cfg.stride_w as _,
              in_grad.as_mut().as_mut_ptr(),
              self.stream.conn().raw_stream().ptr,
          ) };
          out_grad.as_ref().post(&self.stream.conn());
          in_grad.as_ref().post(&self.stream.conn());
        }
        _ => unimplemented!(),
      }
    }
  }
}

pub struct DevicePool2dOperator<S, IoBuf: ?Sized> {
  cfg:      Pool2dOperatorConfig,
  //name:     String,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  h_in:     Vec<f32>,
  h_out:    Vec<f32>,
  pooling:  CudnnPoolingOp,
}

impl<S, IoBuf: ?Sized> DevicePool2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: Pool2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DevicePool2dOperator<S, IoBuf>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    let (in_w, in_h, chan) = cfg.in_dim;
    let (out_w, out_h, _) = cfg.out_dim();
    //println!("DEBUG: pool2d: {:?} {:?}", cfg.in_dim, cfg.out_dim());
    let mut h_in = Vec::with_capacity(cfg.batch_sz * cfg.in_dim.flat_len());
    h_in.resize(cfg.batch_sz * cfg.in_dim.flat_len(), 0.0);
    let mut h_out = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    h_out.resize(cfg.batch_sz * cfg.out_dim().flat_len(), 0.0);
    let pooling = match CudnnPoolingOp::create_2d(
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, chan, cfg.batch_sz).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, chan, cfg.batch_sz).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, chan, cfg.batch_sz).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, chan, cfg.batch_sz).unwrap(),
        cfg.pool_w,   cfg.pool_h,
        cfg.stride_w, cfg.stride_h,
        cfg.pad_w,    cfg.pad_h,
        match cfg.kind {
          PoolKind::Max     => cudnnPoolingMode_t::Max,
          PoolKind::Average => cudnnPoolingMode_t::AverageCountIncludingPadding,
          //PoolKind::Average => cudnnPoolingMode_t::AverageCountExcludingPadding,
        },
    ) {
      Ok(pooling) => pooling,
      Err(e) => panic!("failed to create CudnnPoolingOp: {:?}", e),
    };
    Rc::new(RefCell::new(DevicePool2dOperator{
      cfg:      cfg,
      //name:     String::new(),
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap, stream.clone()),
      h_in:     h_in,
      h_out:    h_out,
      pooling:  pooling,
    }))
  }

  /*pub fn set_name(&mut self, name: &str) {
    self.name = String::from(name);
  }*/
}

impl<S, IoBuf: ?Sized> Operator for DevicePool2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> DeviceOperator for DevicePool2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for DevicePool2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DevicePool2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for DevicePool2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);
    let in_buf = self.in_.buf.borrow();
    let mut out_buf = self.out.buf.borrow_mut();

    in_buf.as_ref().wait(&self.stream.conn());
    out_buf.as_ref().wait(&self.stream.conn());
    self.pooling.set_batch_size(batch_size).unwrap();
    unsafe { self.pooling.forward(
        in_buf.as_ptr(),
        out_buf.as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ) }.unwrap();
    in_buf.as_ref().post(&self.stream.conn());
    out_buf.as_ref().post(&self.stream.conn());

    /*in_buf.as_ref().store_sync(&mut self.h_in, self.stream.conn());
    out_buf.as_ref().store_sync(&mut self.h_out, self.stream.conn());
    dump_to_file(&PathBuf::from(&format!("{}_in.bin", self.name)), &self.h_in);
    dump_to_file(&PathBuf::from(&format!("{}_out.bin", self.name)), &self.h_out);*/
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let in_buf = self.in_.buf.borrow();
      let out_buf = self.out.buf.borrow();
      let out_grad = self.out.grad.as_ref().unwrap().borrow();
      let mut in_grad = in_grad.borrow_mut();

      in_buf.as_ref().wait(&self.stream.conn());
      out_buf.as_ref().wait(&self.stream.conn());
      out_grad.as_ref().wait(&self.stream.conn());
      in_grad.as_ref().wait(&self.stream.conn());
      self.pooling.set_batch_size(batch_size).unwrap();
      unsafe { self.pooling.backward(
          in_buf.as_ref().as_ptr(),
          out_buf.as_ref().as_ptr(),
          out_grad.as_ref().as_ptr(),
          in_grad.as_mut().as_mut_ptr(),
          &*self.stream.conn().cudnn(),
      ) }.unwrap();
      in_buf.as_ref().post(&self.stream.conn());
      out_buf.as_ref().post(&self.stream.conn());
      out_grad.as_ref().post(&self.stream.conn());
      in_grad.as_ref().post(&self.stream.conn());
    }
  }

  fn _backward2(&mut self) {
    let batch_size = self.out.batch_sz.get();
    let in_buf = self.in_.buf.borrow();
    let out_buf = self.out.buf.borrow();
    let out_grad2 = self.out.grad2(self.stream.conn());
    let mut in_grad2 = self.in_.grad2(self.stream.conn());

    match self.cfg.kind {
      PoolKind::Average => {
        in_buf.as_ref().wait(&self.stream.conn());
        out_buf.as_ref().wait(&self.stream.conn());
        out_grad2.as_ref().wait(&self.stream.conn());
        in_grad2.as_ref().wait(&self.stream.conn());
        self.pooling.set_batch_size(batch_size).unwrap();
        unsafe { self.pooling.backward(
            in_buf.as_ref().as_ptr(),
            out_buf.as_ref().as_ptr(),
            out_grad2.as_ref().as_ptr(),
            in_grad2.as_mut().as_mut_ptr(),
            &*self.stream.conn().cudnn(),
        ) }.unwrap();
        in_buf.as_ref().post(&self.stream.conn());
        out_buf.as_ref().post(&self.stream.conn());
        out_grad2.as_ref().post(&self.stream.conn());
        in_grad2.as_ref().post(&self.stream.conn());

        let in_len = self.cfg.in_dim.flat_len();
        let pool_size = self.cfg.pool_w * self.cfg.pool_h;
        in_grad2.as_mut().reshape_mut(in_len * batch_size).scale(1.0 / pool_size as f32, self.stream.conn());
      }
      _ => unimplemented!(),
    }
  }
}
