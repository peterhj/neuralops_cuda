use prelude::*;
use kernels::*;

use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use operator::prelude::*;
use operator::io::{IoBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
//use std::cmp::{max};
//use std::marker::{PhantomData};
use std::rc::{Rc};

pub struct DeviceAffineOperator<S> {
  cfg:      AffineOperatorConfig,
  node:     OperatorNode,
  conn:     DeviceConn,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  hweights: Array2d<f32>,
  hbias:    Array1d<f32>,
  weights:  DeviceArray2d<f32>,
  w_grad:   DeviceArray2d<f32>,
  bias:     DeviceArray1d<f32>,
  b_grad:   DeviceArray1d<f32>,
  tmp_buf:  DeviceMem<f32>,
  tmp_grad: DeviceMem<f32>,
  //act_kern: ActivateKernel,
}

impl<S> DeviceAffineOperator<S> {
  pub fn new<InOp>(cfg: AffineOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, conn: DeviceConn) -> Rc<RefCell<DeviceAffineOperator<S>>> where InOp: 'static + DeviceOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let out_len = cfg.batch_sz * cfg.out_dim;
    let in_ = prev_op.borrow()._output(prev_arm);
    let out = DeviceOutput::new(cfg.batch_sz, cfg.out_dim, cap, conn.clone());
    Rc::new(RefCell::new(DeviceAffineOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      conn:     conn.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      out,
      hweights: Array2d::zeros((cfg.in_dim, cfg.out_dim)),
      hbias:    Array1d::zeros(cfg.out_dim),
      weights:  DeviceArray2d::zeros((cfg.in_dim, cfg.out_dim), conn.clone()),
      w_grad:   DeviceArray2d::zeros((cfg.in_dim, cfg.out_dim), conn.clone()),
      bias:     DeviceArray1d::zeros(cfg.out_dim, conn.clone()),
      b_grad:   DeviceArray1d::zeros(cfg.out_dim, conn.clone()),
      tmp_buf:  DeviceMem::zeros(out_len, conn.clone()),
      tmp_grad: DeviceMem::zeros(out_len, conn.clone()),
      //act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim, cfg.act_kind),
    }))
  }
}

impl<S> Operator for DeviceAffineOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> DeviceOperator for DeviceAffineOperator<S> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for DeviceAffineOperator<S> {
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

  fn _diff_param_sz(&self) -> usize {
    self.cfg.in_dim * self.cfg.out_dim + self.cfg.out_dim
  }

  fn _init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    match self.cfg.w_init {
      ParamInitKind::Disabled => {
        panic!("parameter initialization explicitly disabled");
      }
      ParamInitKind::Uniform{lo, hi} => {
        let dist = Range::new(lo, hi);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Xavier => {
        let half_range = (6.0 / (self.cfg.in_dim + self.cfg.out_dim) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim, self.cfg.out_dim) as f64).sqrt();
        let std = (2.0 / self.cfg.in_dim as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    for e in self.hbias.as_mut_slice().iter_mut() {
      *e = 0.0;
    }
    self.weights.as_view_mut().load_sync(self.hweights.as_view(), self.conn.clone());
    self.bias.as_view_mut().load_sync(self.hbias.as_view(), self.conn.clone());
  }

  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read_buf(offset, self.hweights.as_mut_slice());
    offset += param_reader.read_buf(offset, self.hbias.as_mut_slice());
    self.weights.as_view_mut().load_sync(self.hweights.as_view(), self.conn.clone());
    self.bias.as_view_mut().load_sync(self.hbias.as_view(), self.conn.clone());
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    self.weights.as_view().store_sync(self.hweights.as_view_mut(), self.conn.clone());
    self.bias.as_view().store_sync(self.hbias.as_view_mut(), self.conn.clone());
    let mut offset = init_offset;
    offset += param_writer.write_buf(offset, self.hweights.as_slice());
    offset += param_writer.write_buf(offset, self.hbias.as_slice());
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut [f32]) -> usize {
    self.w_grad.as_view().store_sync(self.hweights.as_view_mut(), self.conn.clone());
    self.b_grad.as_view().store_sync(self.hbias.as_view_mut(), self.conn.clone());
    let mut offset = init_offset;
    offset += grad_writer.write_buf(offset, self.hweights.as_slice());
    offset += grad_writer.write_buf(offset, self.hbias.as_slice());
    offset - init_offset
  }

  fn _reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0, self.conn.clone());
    self.b_grad.as_view_mut().set_constant(0.0, self.conn.clone());
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    let in_buf = self.in_.buf.borrow();
    self.tmp_buf.as_mut().reshape_mut((self.cfg.out_dim, batch_size))
      .matrix_prod(
          1.0,
          self.weights.as_view(), Transpose::T,
          in_buf.as_ref().reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          0.0,
          self.conn.clone(),
      );
    for j in 0 .. batch_size {
      self.tmp_buf.as_mut().reshape_mut((self.cfg.out_dim, batch_size))
        .view_mut((0, j), (self.cfg.out_dim, j+1))
        .matrix_add(
            1.0,
            self.bias.as_view().reshape((self.cfg.out_dim, 1)),
            self.conn.clone(),
        );
    }

    // FIXME: activation.
    //self.act_kern.forward(batch_size, &self.tmp_buf, &mut *self.out.buf.borrow_mut());
    let mut out_buf = self.out.buf.borrow_mut();
    self.tmp_buf.as_ref().wait(&self.conn);
    out_buf.as_ref().wait(&self.conn);
    unsafe { neuralops_cuda_activate_rect_fwd(
        self.tmp_buf.as_ref().as_ptr(),
        batch_size * self.cfg.out_dim,
        out_buf.as_mut().as_mut_ptr(),
        self.conn.stream().ptr,
    ) };
    self.tmp_buf.as_ref().post(&self.conn);
    out_buf.as_ref().post(&self.conn);
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    // FIXME: activation.
    //self.act_kern.backward(batch_size, &self.tmp_buf, &self.out.grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    let in_buf = self.in_.buf.borrow();
    self.w_grad.as_view_mut()
      .matrix_prod(
          1.0,
          in_buf.as_ref().reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          self.tmp_grad.as_ref().reshape((self.cfg.out_dim, batch_size)), Transpose::T,
          1.0,
          self.conn.clone(),
      );
    for j in 0 .. batch_size {
      self.b_grad.as_view_mut().reshape_mut((self.cfg.out_dim, 1))
        .matrix_add(
            1.0,
            self.tmp_grad.as_ref().reshape((self.cfg.out_dim, batch_size))
              .view((0, j), (self.cfg.out_dim, j+1)),
            self.conn.clone(),
        );
    }

    if let Some(in_grad) = self.in_.grad.as_ref() {
      let mut in_grad = in_grad.borrow_mut();
      in_grad.as_mut().reshape_mut((self.cfg.in_dim, batch_size))
        .matrix_prod(
            1.0,
            self.weights.as_view(), Transpose::N,
            self.tmp_grad.as_ref().reshape((self.cfg.out_dim, batch_size)), Transpose::N,
            0.0,
            self.conn.clone(),
        );
    }
  }
}
