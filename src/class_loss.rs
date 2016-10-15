use prelude::*;
use softmax::{DeviceSoftmaxKernel};

use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use float::ord::{F32InfNan};
use iter_utils::{argmax};
use neuralops::prelude::*;
use operator::prelude::*;
use operator::io::{IoBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use std::u32;
use std::cell::{RefCell};
//use std::marker::{PhantomData};
use std::rc::{Rc};

pub struct DeviceSoftmaxNLLClassLoss<S> where S: SampleLabel {
  cfg:      ClassLossConfig,
  node:     OperatorNode,
  conn:     DeviceConn,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  batch_nr: Option<usize>,
  losses:   DeviceMem<f32>,
  probs:    DeviceMem<f32>,
  hats:     DeviceMem<u32>,
  acc_loss: f32,
  reg_loss: f32,
  accuracy: usize,
  labels:   DeviceMem<u32>,
  weights:  DeviceMem<f32>,
  targets:  DeviceMem<f32>,
  labels_h: Vec<u32>,
  ws_h:     Vec<f32>,
  ts_h:     Vec<f32>,
  hats_h:   Vec<u32>,
  losses_h: Vec<f32>,
  softmax:  DeviceSoftmaxKernel,
}

impl<S> DeviceSoftmaxNLLClassLoss<S> where S: SampleLabel {
  pub fn new<InOp>(cfg: ClassLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, conn: DeviceConn) -> Rc<RefCell<DeviceSoftmaxNLLClassLoss<S>>> where InOp: 'static + DeviceOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let in_ = prev_op.borrow()._output(prev_arm);
    let mut weights = DeviceMem::zeros(cfg.batch_sz, conn.clone());
    weights.as_mut().set_constant(1.0, conn.clone());
    let mut targets = DeviceMem::zeros(cfg.batch_sz, conn.clone());
    targets.as_mut().set_constant(1.0, conn.clone());
    let mut labels_h = Vec::with_capacity(cfg.batch_sz);
    labels_h.resize(cfg.batch_sz, 0);
    let mut ws_h = Vec::with_capacity(cfg.batch_sz);
    ws_h.resize(cfg.batch_sz, 1.0);
    let mut ts_h = Vec::with_capacity(cfg.batch_sz);
    ts_h.resize(cfg.batch_sz, 1.0);
    let mut hats_h = Vec::with_capacity(cfg.batch_sz);
    hats_h.resize(cfg.batch_sz, 0);
    let mut losses_h = Vec::with_capacity(cfg.batch_sz);
    losses_h.resize(cfg.batch_sz, 0.0);
    Rc::new(RefCell::new(DeviceSoftmaxNLLClassLoss{
      cfg:      cfg,
      node:     OperatorNode::default(),
      conn:     conn.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, 1, cap, conn.clone()),
      batch_nr: None,
      losses:   DeviceMem::zeros(cfg.batch_sz, conn.clone()),
      probs:    DeviceMem::zeros(cfg.batch_sz * cfg.num_classes, conn.clone()),
      hats:     DeviceMem::zeros(cfg.batch_sz, conn.clone()),
      acc_loss: 0.0,
      reg_loss: 0.0,
      accuracy: 0,
      labels:   DeviceMem::zeros(cfg.batch_sz, conn.clone()),
      weights:  weights,
      targets:  targets,
      labels_h: labels_h,
      ws_h:     ws_h,
      ts_h:     ts_h,
      hats_h:   hats_h,
      losses_h: losses_h,
      softmax:  DeviceSoftmaxKernel::new(cfg.batch_sz, cfg.num_classes, conn),
    }))
  }
}

impl<S> Operator for DeviceSoftmaxNLLClassLoss<S> where S: SampleLabel {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> DeviceOperator for DeviceSoftmaxNLLClassLoss<S> where S: SampleLabel {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for DeviceSoftmaxNLLClassLoss<S> where S: SampleLabel {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
  }

  fn _next_iteration(&mut self) {
    self.batch_nr = None;
  }

  fn _load_batch(&mut self, samples: &[S]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if let Some(cat) = sample.class() {
        assert!(cat < self.cfg.num_classes as u32);
        assert!(cat != u32::MAX);
        self.labels_h[idx] = cat;
      } else {
        self.labels_h[idx] = u32::MAX;
      }
      // FIXME(20161013): sample trait bounds.
      //self.weights[idx] = 1.0;
      //self.weights[idx] = sample.weight().unwrap_or(1.0);
    }
    self.labels.as_mut().load_sync(&self.labels_h, self.conn.clone());
    self.out.batch_sz.set(actual_batch_size);
    self.batch_nr = Some(self.batch_nr.map_or(0, |batch| batch + 1));
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert_eq!(batch_size, self.out.batch_sz.get());

    let in_buf = self.in_.buf.borrow();
    let mut out_buf = self.out.buf.borrow_mut();
    self.softmax._forward(
        batch_size,
        in_buf.as_ref(),
        self.labels.as_ref(),
        self.weights.as_ref(),
        self.targets.as_ref(),
        self.hats.as_mut(),
        self.probs.as_mut(),
        out_buf.as_mut(),
        self.conn.clone(),
    );

    out_buf.as_ref().store_sync(&mut self.losses_h, self.conn.clone());
    self.hats.as_ref().store_sync(&mut self.hats_h, self.conn.clone());
    let mut batch_loss = 0.0;
    let mut batch_accuracy = 0;
    for idx in 0 .. batch_size {
      batch_loss += self.losses_h[idx];
      if self.hats_h[idx] == self.labels_h[idx] {
        batch_accuracy += 1;
      }
    }
    self.acc_loss += batch_loss;
    self.accuracy += batch_accuracy;
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let mut in_grad = in_grad.borrow_mut();
      self.softmax._backward(
          batch_size,
          self.probs.as_ref(),
          self.labels.as_ref(),
          self.weights.as_ref(),
          self.targets.as_ref(),
          in_grad.as_mut(),
          self.conn.clone(),
      );
    }
  }
}

impl<S> DiffLoss<S> for DeviceSoftmaxNLLClassLoss<S> where S: SampleLabel {
  fn reset_loss(&mut self) {
    self.acc_loss = 0.0;
    self.reg_loss = 0.0;
    self.accuracy = 0;
  }

  fn store_loss(&mut self) -> f32 {
    self.acc_loss + self.reg_loss
  }

  fn _store_accuracy(&mut self) -> usize {
    self.accuracy
  }
}
