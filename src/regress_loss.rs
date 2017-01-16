use prelude::*;
use kernels::*;
//use softmax::{DeviceSoftmaxKernel};

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

pub struct DeviceLstSqLoss<S, IoBuf: ?Sized> {
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  loss:     DeviceMem<f32>,
  r_loss:   DeviceMem<f32>,
  target:   DeviceMem<f32>,
  weight:   DeviceMem<u32>,
  jac_targ: DeviceMem<f32>,
  h_target: AsyncMem<f32>,
  h_weight: AsyncMem<f32>,
}

pub struct DeviceIndLstSqRegressLoss<S, IoBuf: ?Sized> {
  cfg:      IndLstSqRegressLossConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  batch_nr: Option<usize>,
  losses:   DeviceMem<f32>,
  //probs:    DeviceMem<f32>,
  //hats:     DeviceMem<u32>,
  nsamples: usize,
  acc_loss: f32,
  reg_loss: f32,
  //accuracy: usize,
  targets:  DeviceMem<f32>,
  labels:   DeviceMem<u32>,
  weights:  DeviceMem<f32>,
  _unused:  DeviceMem<f32>,
  htargets: Vec<f32>,
  labels_h: Vec<u32>,
  ws_h:     Vec<f32>,
  //ts_h:     Vec<f32>,
  //hats_h:   Vec<u32>,
  preds_h:  Vec<f32>,
  delta_h:  Vec<f32>,
  losses_h: Vec<f32>,
  //softmax:  DeviceSoftmaxKernel,
}

impl<S, IoBuf: ?Sized> DeviceIndLstSqRegressLoss<S, IoBuf> {
  pub fn new<InOp>(cfg: IndLstSqRegressLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceIndLstSqRegressLoss<S, IoBuf>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    let mut weights = DeviceMem::zeros(cfg.batch_sz, stream.conn());
    weights.as_mut().set_constant(1.0, stream.conn());
    let mut _unused = DeviceMem::zeros(cfg.batch_sz, stream.conn());
    _unused.as_mut().set_constant(1.0, stream.conn());
    let mut htargets = Vec::with_capacity(cfg.batch_sz);
    htargets.resize(cfg.batch_sz, 0.0);
    let mut labels_h = Vec::with_capacity(cfg.batch_sz);
    labels_h.resize(cfg.batch_sz, 0);
    let mut ws_h = Vec::with_capacity(cfg.batch_sz);
    ws_h.resize(cfg.batch_sz, 1.0);
    //let mut ts_h = Vec::with_capacity(cfg.batch_sz);
    //ts_h.resize(cfg.batch_sz, 1.0);
    //let mut hats_h = Vec::with_capacity(cfg.batch_sz);
    //hats_h.resize(cfg.batch_sz, 0);
    let mut preds_h = Vec::with_capacity(cfg.batch_sz * cfg.index_sz);
    preds_h.resize(cfg.batch_sz * cfg.index_sz, 0.0);
    let mut delta_h = Vec::with_capacity(cfg.batch_sz * cfg.index_sz);
    delta_h.resize(cfg.batch_sz * cfg.index_sz, 0.0);
    let mut losses_h = Vec::with_capacity(cfg.batch_sz);
    losses_h.resize(cfg.batch_sz, 0.0);
    Rc::new(RefCell::new(DeviceIndLstSqRegressLoss{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, 1, cap, stream.conn()),
      batch_nr: None,
      losses:   DeviceMem::zeros(cfg.batch_sz, stream.conn()),
      //probs:    DeviceMem::zeros(cfg.batch_sz * cfg.index_sz, stream.conn()),
      //hats:     DeviceMem::zeros(cfg.batch_sz, stream.conn()),
      nsamples: 0,
      acc_loss: 0.0,
      reg_loss: 0.0,
      //accuracy: 0,
      targets:  DeviceMem::zeros(cfg.batch_sz, stream.conn()),
      labels:   DeviceMem::zeros(cfg.batch_sz, stream.conn()),
      weights:  weights,
      _unused:  _unused,
      htargets: htargets,
      labels_h: labels_h,
      ws_h:     ws_h,
      //ts_h:     ts_h,
      //hats_h:   hats_h,
      preds_h:  preds_h,
      delta_h:  delta_h,
      losses_h: losses_h,
      //softmax:  DeviceSoftmaxKernel::new(cfg.batch_sz, cfg.index_sz, stream.conn()),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for DeviceIndLstSqRegressLoss<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> DeviceOperator for DeviceIndLstSqRegressLoss<S, IoBuf> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<IoBuf: ?Sized> DiffLoss<SampleItem, IoBuf> for DeviceIndLstSqRegressLoss<SampleItem, IoBuf> {
  fn reset_loss(&mut self) {
    self.nsamples = 0;
    self.acc_loss = 0.0;
    self.reg_loss = 0.0;
    //self.accuracy = 0;
  }

  fn store_loss(&mut self) -> f32 {
    self.acc_loss + self.reg_loss
  }

  /*fn _store_accuracy(&mut self) -> usize {
    self.accuracy
  }*/

  fn _get_pred(&mut self) -> &[f32] {
    &self.preds_h
  }

  fn _get_target(&mut self) -> &[f32] {
    &self.htargets
  }

  fn _get_delta(&mut self) -> &[f32] {
    &self.delta_h
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for DeviceIndLstSqRegressLoss<S, IoBuf> {
  default fn _load_batch(&mut self, samples: &[S]) {
    unimplemented!();
  }
}

impl<IoBuf: ?Sized> DiffOperatorData<SampleItem> for DeviceIndLstSqRegressLoss<SampleItem, IoBuf> {
  fn _load_batch(&mut self, samples: &[SampleItem]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if sample.kvs.contains::<SampleRegressTargetKey>() {
        let target = *sample.kvs.get::<SampleRegressTargetKey>().unwrap();
        self.htargets[idx] = target;
      } else {
        self.htargets[idx] = 0.0;
      }
      if sample.kvs.contains::<SampleClassLabelKey>() {
        let cat = *sample.kvs.get::<SampleClassLabelKey>().unwrap();
        assert!(cat < self.cfg.index_sz as u32);
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
    }
    self.targets.as_mut().load_sync(&self.htargets, self.stream.conn());
    self.labels.as_mut().load_sync(&self.labels_h, self.stream.conn());
    self.weights.as_mut().load_sync(&self.ws_h, self.stream.conn());
    self.out.batch_sz.set(actual_batch_size);
    self.batch_nr = Some(self.batch_nr.map_or(0, |batch| batch + 1));
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DeviceIndLstSqRegressLoss<S, IoBuf> {
}

impl<IoBuf: ?Sized> DiffOperator<SampleItem, IoBuf> for DeviceIndLstSqRegressLoss<SampleItem, IoBuf> {
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

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert_eq!(batch_size, self.out.batch_sz.get());

    let in_buf = self.in_.buf.borrow();

    in_buf.as_ref().wait(&self.stream.conn());
    self.targets.as_ref().wait(&self.stream.conn());
    self.labels.as_ref().wait(&self.stream.conn());
    self.weights.as_ref().wait(&self.stream.conn());
    self.losses.as_ref().wait(&self.stream.conn());
    unsafe { neuralops_cuda_ind_lst_sq_fwd(
        in_buf.as_ref().as_ptr(),
        self.cfg.index_sz,
        batch_size,
        self.targets.as_ref().as_ptr(),
        self.labels.as_ref().as_ptr(),
        self.weights.as_ref().as_ptr(),
        self.losses.as_mut().as_mut_ptr(),
        self.stream.conn().raw_stream().ptr,
    ) };
    in_buf.as_ref().post(&self.stream.conn());
    self.targets.as_ref().post(&self.stream.conn());
    self.labels.as_ref().post(&self.stream.conn());
    self.weights.as_ref().post(&self.stream.conn());
    self.losses.as_ref().post(&self.stream.conn());

    in_buf.as_ref().store_sync(&mut self.preds_h, self.stream.conn());
    self.losses.as_ref().store_sync(&mut self.losses_h, self.stream.conn());

    let mut batch_loss = 0.0;
    //let mut batch_accuracy = 0;
    for idx in 0 .. batch_size {
      batch_loss += self.losses_h[idx];
      /*if self.hats_h[idx] == self.labels_h[idx] {
        batch_accuracy += 1;
      }*/
    }
    self.nsamples += batch_size;
    self.acc_loss += batch_loss;
    //self.accuracy += batch_accuracy;
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let in_buf = self.in_.buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      in_buf.as_ref().wait(&self.stream.conn());
      self.targets.as_ref().wait(&self.stream.conn());
      self.labels.as_ref().wait(&self.stream.conn());
      self.weights.as_ref().wait(&self.stream.conn());
      in_grad.as_ref().wait(&self.stream.conn());
      unsafe { neuralops_cuda_ind_lst_sq_bwd(
          in_buf.as_ref().as_ptr(),
          self.cfg.index_sz,
          batch_size,
          self.targets.as_ref().as_ptr(),
          self.labels.as_ref().as_ptr(),
          self.weights.as_ref().as_ptr(),
          in_grad.as_mut().as_mut_ptr(),
          self.stream.conn().raw_stream().ptr,
      ) };
      if let Some(grad_clip) = self.cfg.grad_clip {
        unsafe { neuralops_cuda_clamp(
            in_grad.as_mut().as_mut_ptr(),
            self.cfg.index_sz * batch_size,
            -grad_clip,
            grad_clip,
            self.stream.conn().raw_stream().ptr,
        ) };
      }
      in_buf.as_ref().post(&self.stream.conn());
      self.targets.as_ref().post(&self.stream.conn());
      self.labels.as_ref().post(&self.stream.conn());
      self.weights.as_ref().post(&self.stream.conn());
      in_grad.as_ref().post(&self.stream.conn());

      in_grad.as_ref().store_sync(&mut self.delta_h, self.stream.conn());
    }
  }
}

/*impl<S> LossReport<ClassLossStats> for DeviceIndLstSqRegressLoss<S> {
  fn update_stats(&mut self, iter_nr: usize, stats: &mut ClassLossStats) {
    let batch_size = self.out.batch_sz.get();
    stats.iter_nr = iter_nr;
    stats.sample_count += self.nsamples;
    stats.correct_count += self.accuracy;
    stats.accum_loss += self.acc_loss + self.reg_loss;
  }
}*/
