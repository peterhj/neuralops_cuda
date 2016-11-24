use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use operator::prelude::*;
use operator::opt::sgd::*;

use std::marker::{PhantomData};

#[derive(Clone)]
pub struct DeviceSgdConfig {
  pub step_size:    StepSize,
  pub momentum:     Option<GradientMomentum>,
  pub stream:       DeviceStream,
}

pub struct DeviceSgdUpdate<T> where T: Copy {
  cfg:          DeviceSgdConfig,
  grad_sz:      usize,
  stream:       DeviceStream,
  param:        DeviceMem<T>,
  param_saved:  DeviceMem<T>,
  grad:         DeviceMem<T>,
  diff_acc:     DeviceMem<T>,
  //_marker:      PhantomData<fn (Loss, S)>,
}

impl<Loss, S> GradUpdate<f32, Loss, S, DeviceMem<f32>> for DeviceSgdUpdate<f32> where Loss: DiffLoss<S, DeviceMem<f32>> {
  type Cfg = DeviceSgdConfig;

  fn initialize(cfg: DeviceSgdConfig, loss: &mut Loss) -> DeviceSgdUpdate<f32> {
    let grad_sz = loss.diff_param_sz();
    let stream = cfg.stream.clone();
    let mut param = DeviceMem::zeros(grad_sz, stream.conn());
    let mut param_saved = DeviceMem::zeros(grad_sz, stream.conn());
    let mut grad = DeviceMem::zeros(grad_sz, stream.conn());
    let mut diff_acc = DeviceMem::zeros(grad_sz, stream.conn());
    DeviceSgdUpdate{
      cfg:          cfg,
      grad_sz:      grad_sz,
      stream:       stream,
      param:        param,
      param_saved:  param_saved,
      grad:         grad,
      diff_acc:     diff_acc,
      //_marker:      PhantomData,
    }
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
    self.grad.as_mut().reshape_mut(self.grad_sz).div_scalar(minibatch_sz as f32, self.stream.conn());
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
    if let Some(GradientMomentum::HeavyBall(mu)) = self.cfg.momentum {
      self.diff_acc.as_mut().reshape_mut(self.grad_sz).scale(mu, self.stream.conn());
      self.diff_acc.as_mut().reshape_mut(self.grad_sz).add(-step_size, self.grad.as_ref().reshape(self.grad_sz), self.stream.conn());
      self.param.as_mut().reshape_mut(self.grad_sz).add(1.0, self.diff_acc.as_ref().reshape(self.grad_sz), self.stream.conn());
    } else if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      self.diff_acc.as_mut().reshape_mut(self.grad_sz).scale(mu, self.stream.conn());
      self.diff_acc.as_mut().reshape_mut(self.grad_sz).add(-step_size, self.grad.as_ref().reshape(self.grad_sz), self.stream.conn());
      self.param.as_mut().reshape_mut(self.grad_sz).add(1.0, self.diff_acc.as_ref().reshape(self.grad_sz), self.stream.conn());
    } else {
      self.param.as_mut().reshape_mut(self.grad_sz).add(-step_size, self.grad.as_ref().reshape(self.grad_sz), self.stream.conn());
    }
    loss.load_diff_param(&mut self.param);
  }
}
