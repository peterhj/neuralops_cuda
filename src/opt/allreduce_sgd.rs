use opt::sgd::*;

use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use devicemem_cuda::coll::*;
use operator::prelude::*;
use operator::opt::sgd::*;
use rng::{RngState};
use rng::xorshift::*;

use rand::{Rng, thread_rng};
use std::marker::{PhantomData};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct DeviceAllreduceSgdUpdateBuilder {
  num_workers:  usize,
  rng_state:    [u64; 2],
  comm:         Arc<Mutex<Option<DeviceNcclCommBuilder>>>,
}

impl DeviceAllreduceSgdUpdateBuilder {
  pub fn new(num_workers: usize) -> Self {
    DeviceAllreduceSgdUpdateBuilder{
      num_workers:  num_workers,
      rng_state:    [thread_rng().next_u64(), thread_rng().next_u64()],
      comm:         Arc::new(Mutex::new(None)),
    }
  }

//impl<Loss, S> ParallelGradUpdateBuilder<f32, Loss, S> for DeviceAllreduceSgdUpdate<f32, Loss, S> where Loss: DiffLoss<S, IoBuf=DeviceMem<f32>> {
  pub fn into_update<T, Loss, S>(self, worker_rank: usize, cfg: SgdConfig, stream: DeviceStream, loss: &mut Loss) -> DeviceAllreduceSgdUpdate<T>
  where T: ZeroBits + Copy, Loss: DiffLoss<S, DeviceMem<T>>,
  {
    let grad_sz = loss.diff_param_sz();
    //let stream = cfg.stream.clone();
    let comm = {
      let comm_builder = {
        let mut comm_builder = self.comm.lock().unwrap();
        if comm_builder.is_none() {
          *comm_builder = Some(DeviceNcclCommBuilder::new(self.num_workers));
        }
        comm_builder.as_ref().unwrap().clone()
      };
      comm_builder.into_worker(worker_rank)
    };
    let mut param = DeviceMem::zeros(grad_sz, stream.conn());
    let mut param_saved = DeviceMem::zeros(grad_sz, stream.conn());
    let mut grad = DeviceMem::zeros(grad_sz, stream.conn());
    let mut diff_acc = DeviceMem::zeros(grad_sz, stream.conn());
    let mut shared_rng = Xorshiftplus128Rng::new(&mut thread_rng());
    shared_rng.set_state(&self.rng_state);
    DeviceAllreduceSgdUpdate{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      cfg:          cfg,
      grad_sz:      grad_sz,
      shared_rng:   shared_rng,
      stream:       stream,
      param:        param,
      param_saved:  param_saved,
      grad:         grad,
      diff_acc:     diff_acc,
      comm:         comm,
      //_marker:      PhantomData,
    }
  }
}

pub struct DeviceAllreduceSgdUpdate<T> where T: Copy {
  worker_rank:  usize,
  num_workers:  usize,
  cfg:          SgdConfig,
  grad_sz:      usize,
  shared_rng:   Xorshiftplus128Rng,
  stream:       DeviceStream,
  param:        DeviceMem<T>,
  param_saved:  DeviceMem<T>,
  grad:         DeviceMem<T>,
  diff_acc:     DeviceMem<T>,
  comm:         DeviceNcclCommWorker,
  //_marker:      PhantomData<fn (Loss, S)>,
}

//impl DeviceAllreduceSgdUpdate<f32> where Loss: DiffLoss<S, DeviceMem<f32>> {
impl<Loss, S> GradUpdate<f32, Loss, S, DeviceMem<f32>> for DeviceAllreduceSgdUpdate<f32> where Loss: DiffLoss<S, DeviceMem<f32>> {
  type Cfg = DeviceSgdConfig;

  fn initialize(cfg: Self::Cfg, loss: &mut Loss) -> Self {
    unimplemented!();
  }

  fn reset(&mut self, loss: &mut Loss, rng: &mut Xorshiftplus128Rng) {
    //println!("DEBUG: allreduce sgd: rank: {} initializing param...", self.worker_rank);
    loss.init_param(&mut self.shared_rng);
  }

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
    self.comm.allreduce_sum(self.grad.as_mut(), self.stream.conn());
    self.grad.as_mut().reshape_mut(self.grad_sz).div_scalar((minibatch_sz * self.num_workers) as f32, self.stream.conn());
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
    match self.cfg.momentum {
      Some(GradientMomentum::HeavyBall(mu)) |
      Some(GradientMomentum::Nesterov(mu)) => {
        //let mu = self.cfg.momentum.unwrap().mu();
        self.diff_acc.as_mut().reshape_mut(self.grad_sz).scale(mu, self.stream.conn());
        self.diff_acc.as_mut().reshape_mut(self.grad_sz).add(-step_size, self.grad.as_ref().reshape(self.grad_sz), self.stream.conn());
        self.param.as_mut().reshape_mut(self.grad_sz).add(1.0, self.diff_acc.as_ref().reshape(self.grad_sz), self.stream.conn());
      }
      None => {
        self.param.as_mut().reshape_mut(self.grad_sz).add(-step_size, self.grad.as_ref().reshape(self.grad_sz), self.stream.conn());
      }
    }
    loss.load_diff_param(&mut self.param);
  }
}
