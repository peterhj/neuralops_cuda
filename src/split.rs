use prelude::*;

use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use operator::prelude::*;
use operator::io::{IoBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use std::cell::{RefCell};
use std::rc::{Rc};

pub struct DeviceCopySplitOperator<S> {
  cfg:      SplitOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      DeviceOutput,
  out:      Vec<DeviceOutput>,
}

impl<S> DeviceCopySplitOperator<S> {
  pub fn new<InOp>(cfg: SplitOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceCopySplitOperator<S>>> where InOp: 'static + DeviceOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let prev_out = prev_op.borrow()._output(prev_arm);
    let mut out = Vec::with_capacity(cfg.out_arms);
    for arm in 0 .. cfg.out_arms {
      out.push(DeviceOutput::new(cfg.batch_sz, cfg.dim, cap, stream.conn()));
    }
    Rc::new(RefCell::new(DeviceCopySplitOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream,
      in_op:    prev_op,
      in_:      prev_out,
      out:      out,
    }))
  }
}

impl<S> Operator for DeviceCopySplitOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> DeviceOperator for DeviceCopySplitOperator<S> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    self.out[arm].clone()
  }
}

impl<S> NewDiffOperator<S> for DeviceCopySplitOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(self.cfg.out_arms as _));
    if self.node.count() == 1 {
      self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
      apply(self);
    } else if self.node.count() == self.cfg.out_arms as _ {
      for _ in 0 .. self.cfg.out_arms as _ {
        self.node.pop(epoch);
      }
    }
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(self.cfg.out_arms as _));
    if self.node.count() == self.cfg.out_arms as _ {
      apply(self);
      self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
      for _ in 0 .. self.cfg.out_arms as _ {
        self.node.pop(epoch);
      }
    }
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert!(batch_size <= self.cfg.batch_sz);
    for arm in 0 .. self.cfg.out_arms {
      self.out[arm].batch_sz.set(batch_size);
      let in_buf = self.in_.buf.borrow();
      let mut out_buf = self.out[arm].buf.borrow_mut();
      out_buf.as_mut().slice_mut(0, batch_size * self.cfg.dim)
        .copy(in_buf.as_ref().slice(0, batch_size * self.cfg.dim), self.stream.conn());
    }
  }

  fn _backward(&mut self) {
    if let Some(in_grad) = self.in_.grad.as_ref() {
      let batch_size = self.out[0].batch_sz.get();
      let mut in_grad = in_grad.borrow_mut();
      {
        let out_grad = self.out[0].grad.as_ref().unwrap().borrow();
        in_grad.as_mut().slice_mut(0, batch_size * self.cfg.dim)
          .copy(out_grad.as_ref().slice(0, batch_size * self.cfg.dim), self.stream.conn());
      }
      for arm in 1 .. self.cfg.out_arms {
        let arm_batch_size = self.out[arm].batch_sz.get();
        assert_eq!(batch_size, arm_batch_size);
        let out_grad = self.out[arm].grad.as_ref().unwrap().borrow();
        in_grad.as_mut().reshape_mut(batch_size * self.cfg.dim)
          .add(1.0, out_grad.as_ref().reshape(batch_size * self.cfg.dim), self.stream.conn());
      }
    }
  }
}
