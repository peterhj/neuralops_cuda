use prelude::*;
use kernels::*;
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

pub struct DeviceSoftmaxNLLClassLoss<S, IoBuf: ?Sized> /*where S: SampleLabel*/ {
  cfg:      ClassLossConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  batch_nr: Option<usize>,
  losses:   DeviceMem<f32>,
  probs:    DeviceMem<f32>,
  hats:     DeviceMem<u32>,
  nsamples: usize,
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
  probs_h:  Vec<f32>,
  losses_h: Vec<f32>,
  softmax:  DeviceSoftmaxKernel,
}

impl<S, IoBuf: ?Sized> DeviceSoftmaxNLLClassLoss<S, IoBuf> /*where S: SampleLabel*/ {
  pub fn new<InOp>(cfg: ClassLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceSoftmaxNLLClassLoss<S, IoBuf>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    let mut weights = DeviceMem::zeros(cfg.batch_sz, stream.conn());
    weights.as_mut().set_constant(1.0, stream.conn());
    let mut targets = DeviceMem::zeros(cfg.batch_sz, stream.conn());
    targets.as_mut().set_constant(1.0, stream.conn());
    let mut labels_h = Vec::with_capacity(cfg.batch_sz);
    labels_h.resize(cfg.batch_sz, 0);
    let mut ws_h = Vec::with_capacity(cfg.batch_sz);
    ws_h.resize(cfg.batch_sz, 1.0);
    let mut ts_h = Vec::with_capacity(cfg.batch_sz);
    ts_h.resize(cfg.batch_sz, 1.0);
    let mut hats_h = Vec::with_capacity(cfg.batch_sz);
    hats_h.resize(cfg.batch_sz, 0);
    let mut probs_h = Vec::with_capacity(cfg.batch_sz * cfg.num_classes);
    probs_h.resize(cfg.batch_sz * cfg.num_classes, 0.0);
    let mut losses_h = Vec::with_capacity(cfg.batch_sz);
    losses_h.resize(cfg.batch_sz, 0.0);
    Rc::new(RefCell::new(DeviceSoftmaxNLLClassLoss{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, 1, cap, stream.conn()),
      batch_nr: None,
      losses:   DeviceMem::zeros(cfg.batch_sz, stream.conn()),
      probs:    DeviceMem::zeros(cfg.batch_sz * cfg.num_classes, stream.conn()),
      hats:     DeviceMem::zeros(cfg.batch_sz, stream.conn()),
      nsamples: 0,
      acc_loss: 0.0,
      reg_loss: 0.0,
      accuracy: 0,
      labels:   DeviceMem::zeros(cfg.batch_sz, stream.conn()),
      weights:  weights,
      targets:  targets,
      labels_h: labels_h,
      ws_h:     ws_h,
      ts_h:     ts_h,
      hats_h:   hats_h,
      probs_h:  probs_h,
      losses_h: losses_h,
      softmax:  DeviceSoftmaxKernel::new(cfg.batch_sz, cfg.num_classes, stream.conn()),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for DeviceSoftmaxNLLClassLoss<S, IoBuf> /*where S: SampleLabel*/ {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> DeviceOperator for DeviceSoftmaxNLLClassLoss<S, IoBuf> /*where S: SampleLabel*/ {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<IoBuf: ?Sized> DiffLoss<SampleItem, IoBuf> for DeviceSoftmaxNLLClassLoss<SampleItem, IoBuf> {
  fn reset_loss(&mut self) {
    self.nsamples = 0;
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

  fn _get_pred(&mut self) -> &[f32] {
    &self.probs_h
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DeviceSoftmaxNLLClassLoss<S, IoBuf> {
}

impl<IoBuf: ?Sized> DiffOperator<SampleItem, IoBuf> for DeviceSoftmaxNLLClassLoss<SampleItem, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<SampleItem, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<SampleItem, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    self.node.pop(epoch);
  }

  fn _next_iteration(&mut self) {
    self.batch_nr = None;
    /*self.nsamples = 0;
    self.acc_loss = 0;
    self.accuracy = 0;*/
  }

  fn _load_batch(&mut self, samples: &[SampleItem]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if sample.kvs.contains::<SampleClassLabelKey>() {
        let cat = *sample.kvs.get::<SampleClassLabelKey>().unwrap();
        assert!(cat < self.cfg.num_classes as u32);
        assert!(cat != u32::MAX);
        self.labels_h[idx] = cat;
      } else {
        self.labels_h[idx] = u32::MAX;
      }
      if sample.kvs.contains::<SampleWeightKey>() {
        let weight = *sample.kvs.get::<SampleWeightKey>().unwrap();
        self.ws_h[idx] = weight;
      } else {
        self.ws_h[idx] = 1.0;
      }
      /*if let Some(cat) = sample.class() {
        assert!(cat < self.cfg.num_classes as u32);
        assert!(cat != u32::MAX);
        self.labels_h[idx] = cat;
      } else {
        self.labels_h[idx] = u32::MAX;
      }
      // FIXME(20161013): sample trait bounds.
      //self.weights[idx] = 1.0;
      //self.weights[idx] = sample.weight().unwrap_or(1.0);*/
    }
    self.labels.as_mut().load_sync(&self.labels_h, self.stream.conn());
    self.weights.as_mut().load_sync(&self.ws_h, self.stream.conn());
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
        self.stream.conn(),
    );

    out_buf.as_ref().store_sync(&mut self.losses_h, self.stream.conn());
    self.hats.as_ref().store_sync(&mut self.hats_h, self.stream.conn());
    self.probs.as_ref().store_sync(&mut self.probs_h, self.stream.conn());
    let mut batch_loss = 0.0;
    let mut batch_accuracy = 0;
    for idx in 0 .. batch_size {
      batch_loss += self.losses_h[idx];
      if self.hats_h[idx] == self.labels_h[idx] {
        batch_accuracy += 1;
      }
    }
    self.nsamples += batch_size;
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
          self.stream.conn(),
      );
    }
  }
}

impl<S, IoBuf: ?Sized> LossReport<ClassLossStats> for DeviceSoftmaxNLLClassLoss<S, IoBuf> {
  fn update_stats(&mut self, iter_nr: usize, stats: &mut ClassLossStats) {
    let batch_size = self.out.batch_sz.get();
    stats.iter_nr = iter_nr;
    stats.sample_count += self.nsamples;
    stats.correct_count += self.accuracy;
    stats.accum_loss += self.acc_loss + self.reg_loss;
  }
}

pub struct DeviceLogisticNLLClassLoss<S, IoBuf: ?Sized> {
  cfg:      BinaryClassLossConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  batch_nr: Option<usize>,
  nsamples: usize,
  acc_loss: f32,
  reg_loss: f32,
  accuracy: usize,
  losses:   DeviceMem<f32>,
  hats:     DeviceMem<f32>,
  labels:   DeviceMem<u32>,
  weights:  DeviceMem<f32>,
  losses_h: Vec<f32>,
  labels_h: Vec<u32>,
  ws_h:     Vec<f32>,
  hats_h:   Vec<f32>,
}

impl<S, IoBuf: ?Sized> DeviceLogisticNLLClassLoss<S, IoBuf> {
  pub fn new<InOp>(cfg: BinaryClassLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceLogisticNLLClassLoss<S, IoBuf>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    let mut weights = DeviceMem::zeros(cfg.batch_sz, stream.conn());
    weights.as_mut().set_constant(1.0, stream.conn());
    let mut labels_h = Vec::with_capacity(cfg.batch_sz);
    labels_h.resize(cfg.batch_sz, 0);
    let mut ws_h = Vec::with_capacity(cfg.batch_sz);
    ws_h.resize(cfg.batch_sz, 1.0);
    let mut hats_h = Vec::with_capacity(cfg.batch_sz);
    hats_h.resize(cfg.batch_sz, 0.0);
    let mut losses_h = Vec::with_capacity(cfg.batch_sz);
    losses_h.resize(cfg.batch_sz, 0.0);
    Rc::new(RefCell::new(DeviceLogisticNLLClassLoss{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, 1, cap, stream.conn()),
      batch_nr: None,
      nsamples: 0,
      acc_loss: 0.0,
      reg_loss: 0.0,
      accuracy: 0,
      losses:   DeviceMem::zeros(cfg.batch_sz, stream.conn()),
      hats:     DeviceMem::zeros(cfg.batch_sz, stream.conn()),
      labels:   DeviceMem::zeros(cfg.batch_sz, stream.conn()),
      weights:  weights,
      losses_h: losses_h,
      labels_h: labels_h,
      ws_h:     ws_h,
      hats_h:   hats_h,
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for DeviceLogisticNLLClassLoss<S, IoBuf> /*where S: SampleLabel*/ {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> DeviceOperator for DeviceLogisticNLLClassLoss<S, IoBuf> /*where S: SampleLabel*/ {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<IoBuf: ?Sized> DiffLoss<SampleItem, IoBuf> for DeviceLogisticNLLClassLoss<SampleItem, IoBuf> {
  fn reset_loss(&mut self) {
    self.nsamples = 0;
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

  fn _get_pred(&mut self) -> &[f32] {
    &self.hats_h
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DeviceLogisticNLLClassLoss<S, IoBuf> {
}

impl<IoBuf: ?Sized> DiffOperator<SampleItem, IoBuf> for DeviceLogisticNLLClassLoss<SampleItem, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<SampleItem, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<SampleItem, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    self.node.pop(epoch);
  }

  fn _next_iteration(&mut self) {
    self.batch_nr = None;
  }

  fn _load_batch(&mut self, samples: &[SampleItem]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if sample.kvs.contains::<SampleClassLabelKey>() {
        let cat = *sample.kvs.get::<SampleClassLabelKey>().unwrap();
        assert!(cat < 2);
        self.labels_h[idx] = cat;
      } else {
        self.labels_h[idx] = u32::MAX;
      }
      if sample.kvs.contains::<SampleWeightKey>() {
        let weight = *sample.kvs.get::<SampleWeightKey>().unwrap();
        self.ws_h[idx] = weight;
      } else {
        self.ws_h[idx] = 1.0;
      }
    }
    self.labels.as_mut().load_sync(&self.labels_h, self.stream.conn());
    self.weights.as_mut().load_sync(&self.ws_h, self.stream.conn());
    self.out.batch_sz.set(actual_batch_size);
    self.batch_nr = Some(self.batch_nr.map_or(0, |batch| batch + 1));
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert_eq!(batch_size, self.out.batch_sz.get());

    let in_buf = self.in_.buf.borrow();
    let mut out_buf = self.out.buf.borrow_mut();

    in_buf.as_ref().wait(&self.stream.conn());
    self.labels.as_ref().wait(&self.stream.conn());
    self.weights.as_ref().wait(&self.stream.conn());
    self.losses.as_ref().wait(&self.stream.conn());
    self.hats.as_ref().wait(&self.stream.conn());
    unsafe { neuralops_cuda_logistic_nll_loss_fwd(
        in_buf.as_ref().as_ptr(),
        batch_size,
        self.labels.as_ref().as_ptr(),
        self.weights.as_ref().as_ptr(),
        self.losses.as_mut().as_mut_ptr(),
        self.hats.as_mut().as_mut_ptr(),
        self.stream.conn().raw_stream().ptr,
    ) };
    in_buf.as_ref().post(&self.stream.conn());
    self.labels.as_ref().post(&self.stream.conn());
    self.weights.as_ref().post(&self.stream.conn());
    self.losses.as_ref().post(&self.stream.conn());
    self.hats.as_ref().post(&self.stream.conn());

    self.losses.as_ref().store_sync(&mut self.losses_h, self.stream.conn());
    self.hats.as_ref().store_sync(&mut self.hats_h as &mut [f32], self.stream.conn());
    let mut batch_loss = 0.0;
    let mut batch_accuracy = 0;
    for idx in 0 .. batch_size {
      batch_loss += self.losses_h[idx];
      match (self.hats_h[idx] > 0.5, self.labels_h[idx]) {
        (false, 0) | (true, 1) => {
          batch_accuracy += 1;
        }
        (false, 1) | (true, 0) => {}
        _ => {}
      }
    }
    self.nsamples += batch_size;
    self.acc_loss += batch_loss;
    self.accuracy += batch_accuracy;
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let in_buf = self.in_.buf.borrow();
      let mut in_grad = in_grad.borrow_mut();

      in_buf.as_ref().wait(&self.stream.conn());
      self.labels.as_ref().wait(&self.stream.conn());
      self.weights.as_ref().wait(&self.stream.conn());
      in_grad.as_ref().wait(&self.stream.conn());
      unsafe { neuralops_cuda_logistic_nll_loss_bwd(
          in_buf.as_ref().as_ptr(),
          batch_size,
          self.labels.as_ref().as_ptr(),
          self.weights.as_ref().as_ptr(),
          in_grad.as_mut().as_mut_ptr(),
          self.stream.conn().raw_stream().ptr,
      ) };
      in_buf.as_ref().post(&self.stream.conn());
      self.labels.as_ref().post(&self.stream.conn());
      self.weights.as_ref().post(&self.stream.conn());
      in_grad.as_ref().post(&self.stream.conn());
    }
  }
}
