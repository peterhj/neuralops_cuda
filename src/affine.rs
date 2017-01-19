use prelude::*;
use activate::{DeviceActivateKernel};
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
use std::slice::{from_raw_parts_mut};

pub struct DeviceAffineOperator<S, IoBuf: ?Sized> {
  cfg:      AffineOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  hweights: Array2d<f32>,
  hbias:    Array1d<f32>,
  /*weights:  DeviceArray2d<f32>,
  w_grad:   DeviceArray2d<f32>,
  bias:     DeviceArray1d<f32>,
  b_grad:   DeviceArray1d<f32>,*/
  weights:  Rc<ParamBlock<DeviceArray2d<f32>>>,
  bias:     Rc<ParamBlock<DeviceArray1d<f32>>>,
  tmp_buf:  DeviceMem<f32>,
  tmp_grad: DeviceMem<f32>,
  tmp:      Rc<VarBlock<DeviceMem<f32>>>,
  act_kern: DeviceActivateKernel,
}

impl<S, IoBuf: ?Sized> DeviceAffineOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: AffineOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceAffineOperator<S, IoBuf>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    let out = DeviceOutput::new(cfg.batch_sz, cfg.out_dim, cap, stream.clone());
    Rc::new(RefCell::new(DeviceAffineOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      out,
      hweights: Array2d::zeros((cfg.out_dim, cfg.in_dim)),
      //hweights: Array2d::zeros((cfg.in_dim, cfg.out_dim)),
      hbias:    Array1d::zeros(cfg.out_dim),
      /*weights:  DeviceArray2d::zeros((cfg.out_dim, cfg.in_dim), stream.conn()),
      w_grad:   DeviceArray2d::zeros((cfg.out_dim, cfg.in_dim), stream.conn()),
      //weights:  DeviceArray2d::zeros((cfg.in_dim, cfg.out_dim), stream.conn()),
      //w_grad:   DeviceArray2d::zeros((cfg.in_dim, cfg.out_dim), stream.conn()),
      bias:     DeviceArray1d::zeros(cfg.out_dim, stream.conn()),
      b_grad:   DeviceArray1d::zeros(cfg.out_dim, stream.conn()),*/
      weights:  ParamBlock::new(DefaultParamAllocator::new({ let stream = stream.clone(); move || DeviceArray2d::zeros((cfg.out_dim, cfg.in_dim), stream.conn()) })),
      bias:     ParamBlock::new(DefaultParamAllocator::new({ let stream = stream.clone(); move || DeviceArray1d::zeros(cfg.out_dim, stream.conn()) })),
      tmp_buf:  DeviceMem::zeros(cfg.batch_sz * cfg.out_dim, stream.conn()),
      tmp_grad: DeviceMem::zeros(cfg.batch_sz * cfg.out_dim, stream.conn()),
      tmp:      VarBlock::new(DefaultVarAllocator::new({ let stream = stream.clone(); move || DeviceMem::zeros(cfg.batch_sz * cfg.out_dim, stream.conn()) })),
      act_kern: DeviceActivateKernel::new(cfg.out_dim, cfg.act_kind),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for DeviceAffineOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> DeviceOperator for DeviceAffineOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for DeviceAffineOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DeviceAffineOperator<S, IoBuf> {
  default fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _load_direction(&mut self, init_offset: usize, dir_reader: &mut IoBuf) -> usize {
    unimplemented!();
  }
}

impl<S> DiffOperatorIo<[f32]> for DeviceAffineOperator<S, [f32]> {
  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read_buf(offset, self.hweights.as_mut_slice());
    self.weights.val.as_mut().as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    if self.cfg.bias {
      offset += param_reader.read_buf(offset, self.hbias.as_mut_slice());
      self.bias.val.as_mut().as_view_mut().load_sync(self.hbias.as_view(), self.stream.conn());
    }
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    self.weights.val.as_ref().as_view().store_sync(self.hweights.as_view_mut(), self.stream.conn());
    offset += param_writer.write_buf(offset, self.hweights.as_slice());
    if self.cfg.bias {
      self.bias.val.as_ref().as_view().store_sync(self.hbias.as_view_mut(), self.stream.conn());
      offset += param_writer.write_buf(offset, self.hbias.as_slice());
    }
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    self.weights.grad.as_ref().as_view().store_sync(self.hweights.as_view_mut(), self.stream.conn());
    offset += grad_writer.write_buf(offset, self.hweights.as_slice());
    if self.cfg.bias {
      self.bias.grad.as_ref().as_view().store_sync(self.hbias.as_view_mut(), self.stream.conn());
      offset += grad_writer.write_buf(offset, self.hbias.as_slice());
    }
    offset - init_offset
  }

  fn _load_direction(&mut self, init_offset: usize, dir_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += dir_reader.read_buf(offset, self.hweights.as_mut_slice());
    self.weights.r_dir.as_mut().as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    if self.cfg.bias {
      offset += dir_reader.read_buf(offset, self.hbias.as_mut_slice());
      self.bias.r_dir.as_mut().as_view_mut().load_sync(self.hbias.as_view(), self.stream.conn());
    }
    offset - init_offset
  }
}

impl<S> DiffOperatorIo<DeviceMem<f32>> for DeviceAffineOperator<S, DeviceMem<f32>> {
  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut DeviceMem<f32>) -> usize {
    let mut offset = init_offset;
    let w_len = self.weights.val.as_ref().dim().flat_len();
    self.weights.val.as_mut().as_view_mut().reshape_mut(w_len)
      .copy(param_reader.as_ref().slice(offset, offset + w_len).reshape(w_len), self.stream.conn());
    offset += w_len;
    if self.cfg.bias {
      let b_len = self.bias.val.as_ref().dim().flat_len();
      self.bias.val.as_mut().as_view_mut()
        .copy(param_reader.as_ref().slice(offset, offset + b_len).reshape(b_len), self.stream.conn());
      offset += b_len;
    }
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut DeviceMem<f32>) -> usize {
    let mut offset = init_offset;
    let w_len = self.weights.val.as_ref().dim().flat_len();
    param_writer.as_mut().slice_mut(offset, offset + w_len).reshape_mut(w_len)
      .copy(self.weights.val.as_ref().as_view().reshape(w_len), self.stream.conn());
    offset += w_len;
    if self.cfg.bias {
      let b_len = self.bias.val.as_ref().dim().flat_len();
      param_writer.as_mut().slice_mut(offset, offset + b_len).reshape_mut(b_len)
        .copy(self.bias.val.as_ref().as_view(), self.stream.conn());
      offset += b_len;
    }
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut DeviceMem<f32>) -> usize {
    let mut offset = init_offset;
    let w_len = self.weights.val.as_ref().dim().flat_len();
    grad_writer.as_mut().slice_mut(offset, offset + w_len).reshape_mut(w_len)
      .copy(self.weights.grad.as_ref().as_view().reshape(w_len), self.stream.conn());
    offset += w_len;
    if self.cfg.bias {
      let b_len = self.bias.val.as_ref().dim().flat_len();
      grad_writer.as_mut().slice_mut(offset, offset + b_len).reshape_mut(b_len)
        .copy(self.bias.grad.as_ref().as_view(), self.stream.conn());
      offset += b_len;
    }
    offset - init_offset
  }
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for DeviceAffineOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _diff_param_sz(&self) -> usize {
    if self.cfg.bias {
      self.cfg.in_dim * self.cfg.out_dim + self.cfg.out_dim
    } else {
      self.cfg.in_dim * self.cfg.out_dim
    }
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
        //let half_range = (6.0 / (self.cfg.in_dim + self.cfg.out_dim) as f64).sqrt();
        let half_range = (3.0 / self.cfg.in_dim as f64).sqrt();
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
    self.weights.val.as_mut().as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    self.bias.val.as_mut().as_view_mut().set_constant(0.0, self.stream.conn());
  }

  fn _reset_grad(&mut self) {
    self.weights.grad.as_mut().as_view_mut().set_constant(0.0, self.stream.conn());
    self.bias.grad.as_mut().as_view_mut().set_constant(0.0, self.stream.conn());
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    let in_buf = self.in_.buf.borrow();
    self.tmp_buf.as_mut().reshape_mut((self.cfg.out_dim, batch_size))
      .matrix_prod(
          1.0,
          self.weights.val.as_ref().as_view(), Transpose::N,
          //self.weights.as_view(), Transpose::T,
          in_buf.as_ref().reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          0.0,
          self.stream.conn(),
      );
    if self.cfg.bias {
      /*for j in 0 .. batch_size {
        self.tmp_buf.as_mut().reshape_mut((self.cfg.out_dim, batch_size))
          .view_mut((0, j), (self.cfg.out_dim, j+1))
          .matrix_add(
              1.0,
              self.bias.as_view().reshape((self.cfg.out_dim, 1)),
              self.stream.conn(),
          );
      }*/
      self.tmp_buf.as_ref().wait(&self.stream.conn());
      self.bias.val.as_ref().as_view().wait(&self.stream.conn());
      unsafe { neuralops_cuda_linear_bias_fwd_inplace(
          self.tmp_buf.as_mut().as_mut_ptr(),
          self.cfg.out_dim,
          batch_size,
          self.bias.val.as_ref().as_view().as_ptr(),
          self.stream.conn().raw_stream().ptr,
      ) };
      self.tmp_buf.as_ref().post(&self.stream.conn());
      self.bias.val.as_ref().as_view().post(&self.stream.conn());
    }

    let mut out_buf = self.out.buf.borrow_mut();
    self.act_kern._forward(batch_size, self.tmp_buf.as_ref(), out_buf.as_mut(), self.stream.conn());
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    let out_grad = self.out.grad.as_ref().unwrap().borrow();
    self.act_kern._backward(batch_size, self.tmp_buf.as_ref(), out_grad.as_ref(), self.tmp_grad.as_mut(), self.stream.conn());

    let in_buf = self.in_.buf.borrow();
    self.weights.grad.as_mut().as_view_mut()
      .matrix_prod(
          1.0,
          self.tmp_grad.as_ref().reshape((self.cfg.out_dim, batch_size)), Transpose::N,
          in_buf.as_ref().reshape((self.cfg.in_dim, batch_size)), Transpose::T,
          //in_buf.as_ref().reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          //self.tmp_grad.as_ref().reshape((self.cfg.out_dim, batch_size)), Transpose::T,
          1.0,
          self.stream.conn(),
      );
    if self.cfg.bias {
      /*for j in 0 .. batch_size {
        self.b_grad.as_view_mut().reshape_mut((self.cfg.out_dim, 1))
          .matrix_add(
              1.0,
              self.tmp_grad.as_ref().reshape((self.cfg.out_dim, batch_size))
                .view((0, j), (self.cfg.out_dim, j+1)),
              self.stream.conn(),
          );
      }*/
      self.tmp_grad.as_ref().wait(&self.stream.conn());
      self.bias.grad.as_mut().as_view().wait(&self.stream.conn());
      unsafe { neuralops_cuda_linear_bias_bwd(
          self.tmp_grad.as_ref().as_ptr(),
          self.cfg.out_dim,
          batch_size,
          self.bias.grad.as_mut().as_view_mut().as_mut_ptr(),
          self.stream.conn().raw_stream().ptr,
      ) };
      self.tmp_grad.as_ref().post(&self.stream.conn());
      self.bias.grad.as_mut().as_view().post(&self.stream.conn());
    }

    if let Some(in_grad) = self.in_.grad.as_ref() {
      let mut in_grad = in_grad.borrow_mut();
      in_grad.as_mut().reshape_mut((self.cfg.in_dim, batch_size))
        .matrix_prod(
            1.0,
            self.weights.val.as_ref().as_view(), Transpose::T,
            //self.weights.as_view(), Transpose::N,
            self.tmp_grad.as_ref().reshape((self.cfg.out_dim, batch_size)), Transpose::N,
            0.0,
            self.stream.conn(),
        );
    }
  }

  fn _backward2(&mut self) {
    let batch_size = self.out.batch_sz.get();

    /*
    let out_grad2 = self.out.grad2(self.stream.conn());
    self.act_kern._backward(batch_size, self.tmp_buf.as_ref(), out_grad2.as_ref(), self.tmp_grad.as_mut(), self.stream.conn());

    let in_buf = self.in_.buf.borrow();
    self.w_grad.as_view_mut()
    //self.w_grad2.as_view_mut()
      .matrix_prod(
          1.0,
          self.tmp_grad.as_ref().reshape((self.cfg.out_dim, batch_size)), Transpose::N,
          in_buf.as_ref().reshape((self.cfg.in_dim, batch_size)), Transpose::T,
          1.0,
          self.stream.conn(),
      );
    if self.cfg.bias {
      unimplemented!();
      /*self.tmp_grad.as_ref().wait(&self.stream.conn());
      self.b_grad.as_view().wait(&self.stream.conn());
      unsafe { neuralops_cuda_linear_bias_bwd(
          self.tmp_grad.as_ref().as_ptr(),
          self.cfg.out_dim,
          batch_size,
          self.b_grad.as_view_mut().as_mut_ptr(),
          self.stream.conn().raw_stream().ptr,
      ) };
      self.tmp_grad.as_ref().post(&self.stream.conn());
      self.b_grad.as_view().post(&self.stream.conn());*/
    }

    let mut in_grad2 = self.in_.grad2(self.stream.conn());
    in_grad2.as_mut().reshape_mut((self.cfg.in_dim, batch_size))
      .matrix_prod(
          1.0,
          // FIXME(20170105): square of the weight.
          self.weights.as_view(), Transpose::T,
          self.tmp_grad.as_ref().reshape((self.cfg.out_dim, batch_size)), Transpose::N,
          0.0,
          self.stream.conn(),
      );
    */
  }

  fn _r_forward(&mut self) {
    let batch_size = self.in_.batch_sz.get();
    assert!(batch_size <= self.cfg.batch_sz);

    let in_val = self.in_.buf.borrow();
    self.tmp.r_val.as_mut().as_mut().reshape_mut((self.cfg.out_dim, batch_size))
      .matrix_prod(
          1.0,
          self.weights.val.as_ref().as_view(), Transpose::N,
          self.in_.data.r_val.as_ref().as_ref().reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          0.0,
          self.stream.conn(),
      );
    self.tmp.r_val.as_mut().as_mut().reshape_mut((self.cfg.out_dim, batch_size))
      .matrix_prod(
          1.0,
          self.weights.r_dir.as_ref().as_view(), Transpose::N,
          in_val.as_ref().reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          1.0,
          self.stream.conn(),
      );
    if self.cfg.bias {
      unimplemented!();
    }

    self.act_kern._r_forward(batch_size, self.tmp_buf.as_ref(), self.tmp.r_val.as_ref().as_ref(), self.out.data.r_val.as_mut().as_mut(), self.stream.conn());
  }

  fn _r_backward(&mut self) {
    unimplemented!();
  }

  fn _dump_input(&mut self) -> Vec<u8> {
    let input_sz = self.cfg.batch_sz * self.cfg.in_dim * 4;
    let mut input_data = Vec::with_capacity(input_sz);
    input_data.resize(input_sz, 0);
    {
      let mut input_h = unsafe { from_raw_parts_mut(input_data.as_mut_ptr() as *mut f32, input_data.len() / 4) };
      self.in_.buf.borrow().as_ref().store_sync(&mut input_h, self.stream.conn());
    }
    input_data
  }

  fn _dump_output(&mut self) -> Vec<u8> {
    let output_sz = self.cfg.batch_sz * self.cfg.out_dim * 4;
    let mut output_data = Vec::with_capacity(output_sz);
    output_data.resize(output_sz, 0);
    {
      let mut output_h = unsafe { from_raw_parts_mut(output_data.as_mut_ptr() as *mut f32, output_data.len() / 4) };
      self.tmp_buf.as_ref().store_sync(&mut output_h, self.stream.conn());
    }
    output_data
  }
}
