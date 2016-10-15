use prelude::*;

use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use operator::prelude::*;
use operator::io::{IoBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use std::cell::{RefCell};
use std::rc::{Rc};

pub struct DeviceAddJoinOperator<S> {
  cfg:      JoinOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_ops:   Vec<Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>>,
  in_:      Vec<DeviceOutput>,
  out:      DeviceOutput,
}

impl<S> DeviceAddJoinOperator<S> {
  pub fn new(cfg: JoinOperatorConfig, cap: OpCapability, stream: DeviceStream) -> Rc<RefCell<DeviceAddJoinOperator<S>>> {
    let mut in_ops = Vec::with_capacity(cfg.in_arms);
    let mut in_ = Vec::with_capacity(cfg.in_arms);
    Rc::new(RefCell::new(DeviceAddJoinOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_ops:   in_ops,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, cfg.dim, cap, stream.conn()),
    }))
  }

  pub fn append_input<InOp>(&mut self, in_op: Rc<RefCell<InOp>>, in_arm: usize) where InOp: 'static + DeviceOperator + NewDiffOperator<S, IoBuf=[f32]> {
    assert!(self.in_ops.len() < self.cfg.in_arms);
    assert_eq!(self.in_ops.len(), self.in_.len());
    let out = in_op.borrow()._output(in_arm);
    self.in_ops.push(in_op);
    self.in_.push(out);
  }
}

impl<S> Operator for DeviceAddJoinOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> DeviceOperator for DeviceAddJoinOperator<S> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for DeviceAddJoinOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    for in_op in self.in_ops.iter() {
      in_op.borrow_mut()._traverse_fwd(epoch, apply);
    }
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
    for in_op in self.in_ops.iter() {
      in_op.borrow_mut()._traverse_bwd(epoch, apply);
    }
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_[0].batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);
    let mut out_buf = self.out.buf.borrow_mut();
    {
      let in_buf = self.in_[0].buf.borrow();
      out_buf.as_mut().slice_mut(0, batch_size * self.cfg.dim)
        .copy(in_buf.as_ref().slice(0, batch_size * self.cfg.dim), self.stream.conn());
    }
    for arm in 1 .. self.cfg.in_arms {
      let arm_batch_size = self.in_[arm].batch_sz.get();
      assert_eq!(batch_size, arm_batch_size);
      let in_buf = self.in_[arm].buf.borrow();
      out_buf.as_mut().reshape_mut(batch_size * self.cfg.dim)
        .add(1.0, in_buf.as_ref().reshape(batch_size * self.cfg.dim), 1.0, self.stream.conn());
    }
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    for arm in 0 .. self.cfg.in_arms {
      if let Some(in_grad) = self.in_[arm].grad.as_ref() {
        let out_grad = self.out.grad.as_ref().unwrap().borrow();
        let mut in_grad = in_grad.borrow_mut();
        in_grad.as_mut().slice_mut(0, batch_size * self.cfg.dim)
          .copy(out_grad.as_ref().slice(0, batch_size * self.cfg.dim), self.stream.conn());
      }
    }
  }
}
