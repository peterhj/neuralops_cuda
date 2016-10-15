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
  hats:     DeviceMem<u32>,
  acc_loss: f32,
  reg_loss: f32,
  accuracy: usize,
  labels:   DeviceMem<u32>,
  weights:  DeviceMem<f32>,
  targets:  DeviceMem<f32>,
  softmax:  DeviceSoftmaxKernel,
}

impl<S> DeviceSoftmaxNLLClassLoss<S> where S: SampleLabel {
  pub fn new<InOp>(cfg: ClassLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, conn: DeviceConn) -> Rc<RefCell<DeviceSoftmaxNLLClassLoss<S>>> where InOp: 'static + DeviceOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let mut hats = Vec::with_capacity(cfg.batch_sz);
    hats.resize(cfg.batch_sz, 0);
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    losses.resize(cfg.batch_sz, 0.0);
    let mut labels = Vec::with_capacity(cfg.batch_sz);
    labels.resize(cfg.batch_sz, 0);
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    weights.resize(cfg.batch_sz, 1.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(DeviceSoftmaxNLLClassLoss{
      cfg:      cfg,
      node:     OperatorNode::default(),
      conn:     conn.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, cfg.num_classes, cap, conn.clone()),
      batch_nr: None,
      losses:   DeviceMem::zeros(cfg.batch_sz, conn.clone()),
      hats:     DeviceMem::zeros(cfg.batch_sz, conn.clone()),
      acc_loss: 0.0,
      reg_loss: 0.0,
      accuracy: 0,
      labels:   DeviceMem::zeros(cfg.batch_sz, conn.clone()),
      weights:  DeviceMem::zeros(cfg.batch_sz, conn.clone()),
      targets:  DeviceMem::zeros(cfg.batch_sz, conn.clone()),
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
      // FIXME(20161014)
      if let Some(cat) = sample.class() {
        assert!(cat < self.cfg.num_classes as u32);
        assert!(cat != u32::MAX);
        //self.labels[idx] = cat;
      } else {
        //self.labels[idx] = u32::MAX;
      }
      // FIXME(20161013): sample trait bounds.
      //self.weights[idx] = 1.0;
      //self.weights[idx] = sample.weight().unwrap_or(1.0);
    }
    self.out.batch_sz.set(actual_batch_size);
    self.batch_nr = Some(self.batch_nr.map_or(0, |batch| batch + 1));
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert_eq!(batch_size, self.out.batch_sz.get());

    // FIXME

    /*let in_buf = self.in_.buf.borrow();
    let mut out_buf = self.out.buf.borrow_mut();
    for idx in 0 .. batch_size {
      let range = idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes;
      let max_logit_k = argmax(in_buf[range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      let max_logit = in_buf[idx * self.cfg.num_classes + max_logit_k];
      self.max_log[idx] = max_logit;
      self.hats[idx] = max_logit_k as u32;
      for k in 0 .. self.cfg.num_classes {
        self.facts[idx * self.cfg.num_classes + k] = (in_buf[idx * self.cfg.num_classes + k] - max_logit).exp();
      }
      let sum_fact: f32 = Sum::sum(self.facts[range].iter().map(|&x| x));
      for k in 0 .. self.cfg.num_classes {
        out_buf[idx * self.cfg.num_classes + k] = self.facts[idx * self.cfg.num_classes + k] / sum_fact;
      }
      self.losses[idx] = -self.weights[idx] * out_buf[idx * self.cfg.num_classes + self.labels[idx] as usize].ln();
    }

    let mut batch_loss = 0.0;
    let mut batch_accuracy = 0;
    for idx in 0 .. batch_size {
      let idx_range = idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes;
      let max_logit_k = argmax(in_buf[idx_range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      self.hats[idx] = max_logit_k as u32;
      if self.hats[idx] == self.labels[idx] {
        batch_accuracy += 1;
      }
      let loss = if self.labels[idx] == u32::MAX {
        0.0
      } else {
        -self.weights[idx] * out_buf[idx * self.cfg.num_classes + self.labels[idx] as usize].ln()
      };
      self.losses[idx] = loss;
      batch_loss += loss;
    }
    self.acc_loss += batch_loss;
    self.accuracy += batch_accuracy;

    if let Some(0) = self.batch_nr {
      let mut reg_loss = 0.0;
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        let block_out = block.borrow()._output(0);
        reg_loss += block_out.buf.borrow()[0];
      }*/
      self.reg_loss = reg_loss;
    }*/
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let out_buf = self.out.buf.borrow();
      let mut in_grad = in_grad.borrow_mut();

      // FIXME
      //self.softmax.backward();

      /*let mut p = 0;
      for idx in 0 .. batch_size {
        for k in 0 .. self.cfg.num_classes {
          in_grad[p] =
              self.weights[idx] *
              (out_buf[p] - if k == self.labels[idx] as usize { 1.0 } else { 0.0 });
          p += 1;
        }
      }*/
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
