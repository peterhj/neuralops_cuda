use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use devicemem_cuda::coll::{DeviceRingAllreduce};
use operator::prelude::*;
use operator::opt::sgd::*;

use std::marker::{PhantomData};

#[derive(Clone)]
pub struct DeviceAllreduceSgdUpdateBuilder<T> where T: Copy {
  num_workers:  usize,
  allreduce:    Arc<Mutex<Option<DeviceRingAllreduceBuilder<T>>>>,
}

impl<Loss, S> ParallelGradUpdateBuilder<f32, Loss, S> for DeviceAllreduceSgdUpdate<f32, Loss, S> where Loss: DiffLoss<S, IoBuf=DeviceMem<f32>> {
  fn into_update(self, worker_rank: usize, cfg: DeviceSgdConfig<T>, loss: &mut Loss) -> DeviceAllreduceSgdUpdate<f32, Loss, S> {
    let grad_sz = loss.diff_param_sz();
    let stream = cfg.stream.clone();
    let allreduce = {
      let mut allreduce_builder = self.allreduce.lock().unwrap();
      if allreduce_builder.is_none() {
        *allreduce_builder = Some(DeviceRingAllreduceBuilder::new(self.num_workers, grad_sz));
      }
      allreduce_builder.as_ref().clone().into(worker_rank, stream.clone())
    };
    let mut param = DeviceMem::zeros(grad_sz, stream.conn());
    let mut param_saved = DeviceMem::zeros(grad_sz, stream.conn());
    let mut grad = DeviceMem::zeros(grad_sz, stream.conn());
    let mut diff_acc = DeviceMem::zeros(grad_sz, stream.conn());
    DeviceAllreduceSgdUpdate{
      worker_rank:  worker_rank,
      num_workers:  num_workers,
      cfg:          cfg,
      grad_sz:      grad_sz,
      stream:       stream,
      param:        param,
      param_saved:  param_saved,
      grad:         grad,
      diff_acc:     diff_acc,
      allreduce:    allreduce,
      _marker:      PhantomData,
    }
  }
}

pub struct DeviceAllreduceSgdUpdate<T, Loss, S> where T: Copy {
  worker_rank:  usize,
  num_workers:  usize,
  cfg:          DeviceSgdConfig<T>,
  grad_sz:      usize,
  stream:       DeviceStream,
  param:        DeviceMem<T>,
  param_saved:  DeviceMem<T>,
  grad:         DeviceMem<T>,
  diff_acc:     DeviceMem<T>,
  allreduce:    DeviceRingAllreduce<T>,
  _marker:      PhantomData<fn (Loss, S)>,
}

impl<Loss, S> ParallelGradUpdate<f32, Loss, S> for DeviceAllreduceSgdUpdate<f32, Loss, S> where Loss: DiffLoss<S, IoBuf=DeviceMem<f32>> {
  type Cfg = DeviceSgdConfig<T>;

  fn begin_iteration(&mut self, loss: &mut Loss) {
    if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      loss.store_diff_param(&mut self.param_saved);
      self.param.as_mut().copy(self.param_saved.as_ref(), self.stream.conn());
      self.param.as_mut().reshape_mut(self.grad_sz).add(mu, self.diff_acc.as_ref().reshape(self.grad_sz), self.stream.conn());
      loss.load_diff_param(&mut self.param);
    }
  }

  fn end_iteration(&mut self, minibatch_sz: usize, loss: &mut Loss) {
    if let Some(GradientMomentum::Nesterov(_)) = self.cfg.momentum {
      loss.load_diff_param(&mut self.param_saved);
    }
    loss.store_grad(&mut self.grad);
    self.grad.as_mut().reshape_mut(self.grad_sz).div_scalar(minibatch_sz as f32, self.stream.conn());
    self.allreduce.allreduce(self.grad.as_mut(), self.stream.clone());
  }

  fn step(&mut self, iter_count: usize, loss: &mut Loss) {
    let step_size = match self.cfg.step_size {
      StepSize::Constant(alpha) => {
        alpha
      }
      StepSize::Decay{init_step, step_decay, decay_iters} => {
        let num_decays = iter_count / decay_iters;
        init_step * step_decay.powi(num_decays as i32)
      }
      _ => unimplemented!(),
    };
    loss.store_diff_param(&mut self.param);
    if self.cfg.momentum.is_some() {
      let mu = self.cfg.momentum.unwrap().mu();
      self.diff_acc.as_mut().reshape_mut(self.grad_sz).scale(mu, self.stream.conn());
      self.diff_acc.as_mut().reshape_mut(self.grad_sz).add(-step_size, self.grad.as_ref().reshape(self.grad_sz), self.stream.conn());
      self.param.as_mut().reshape_mut(self.grad_sz).add(1.0, self.diff_acc.as_ref().reshape(self.grad_sz), self.stream.conn());
    } else {
      self.param.as_mut().reshape_mut(self.grad_sz).add(-step_size, self.grad.as_ref().reshape(self.grad_sz), self.stream.conn());
    }
    loss.load_diff_param(&mut self.param);
  }
}
