use prelude::*;
use activate::{DeviceActivateKernel};

use cuda_dnn::v5::{CudnnPoolingOp, CudnnTensorDesc};
use cuda_dnn::v5::ffi::*;
use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use operator::prelude::*;
use operator::io::{IoBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use std::cell::{RefCell};
use std::rc::{Rc};

pub struct DevicePool2dOperator<S> {
  cfg:      Pool2dOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  pooling:  CudnnPoolingOp,
}

impl<S> DevicePool2dOperator<S> {
  pub fn new<InOp>(cfg: Pool2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DevicePool2dOperator<S>>> where InOp: 'static + DeviceOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let in_ = prev_op.borrow()._output(prev_arm);
    let (in_w, in_h, chan) = cfg.in_dim;
    let (out_w, out_h, _) = cfg.out_dim();
    let pooling = match CudnnPoolingOp::create_2d(
        CudnnTensorDesc::<f32>::create_4d(in_w, in_h, chan, cfg.batch_sz).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_w, in_h, chan, cfg.batch_sz).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_w, out_h, chan, cfg.batch_sz).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_w, out_h, chan, cfg.batch_sz).unwrap(),
        cfg.pool_w,   cfg.pool_h,
        cfg.stride_w, cfg.stride_h,
        cfg.pad_w,    cfg.pad_h,
        match cfg.kind {
          PoolKind::Max     => cudnnPoolingMode_t::Max,
          PoolKind::Average => cudnnPoolingMode_t::AverageCountExcludingPadding,
        },
    ) {
      Ok(pooling) => pooling,
      Err(e) => panic!("failed to create CudnnPoolingOp: {:?}", e),
    };
    Rc::new(RefCell::new(DevicePool2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap, stream.conn()),
      pooling:  pooling,
    }))
  }
}

impl<S> Operator for DevicePool2dOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> DeviceOperator for DevicePool2dOperator<S> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for DevicePool2dOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
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
}
