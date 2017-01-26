use prelude::*;
use activate::{DeviceActivateKernel};
use kernels::*;

use cuda_dnn::v5::{CudnnTensorLayout, CudnnTensorDesc, CudnnFilterDesc, CudnnConvDesc, CudnnAddOp, CudnnConvFwdOp, CudnnConvBwdFilterOp, CudnnConvBwdDataOp};
use cuda_dnn::v5::ffi::*;
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
use std::cmp::{max};
use std::rc::{Rc};
use std::slice::{from_raw_parts_mut};

pub struct DeviceConv2dScaleKernel {
  pub dim:      (usize, usize, usize),
  pub h_scale:  Array1d<f32>,
  pub h_bias:   Array1d<f32>,
  pub scale:    Rc<ParamBlock<DeviceArray1d<f32>>>,
  pub bias:     Rc<ParamBlock<DeviceArray1d<f32>>>,
}

impl DeviceConv2dScaleKernel {
  pub fn new(dim: (usize, usize, usize), stream: DeviceStream) -> Self {
    let chan = dim.2;
    DeviceConv2dScaleKernel{
      dim:      dim,
      h_scale:  Array1d::zeros(chan),
      h_bias:   Array1d::zeros(chan),
      scale:    ParamBlock::new(DefaultParamAllocator::new({ let stream = stream.clone(); move || DeviceArray1d::zeros(chan, stream.conn()) })),
      bias:     ParamBlock::new(DefaultParamAllocator::new({ let stream = stream.clone(); move || DeviceArray1d::zeros(chan, stream.conn()) })),
    }
  }

  pub fn _forward<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, mut out_buf: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    in_buf.wait(&conn);
    self.scale.val.as_ref().as_view().wait(&conn);
    self.bias.val.as_ref().as_view().wait(&conn);
    out_buf.wait(&conn);
    unsafe { neuralops_cuda_conv2d_scale_fwd(
        in_buf.as_ptr(),
        self.dim.0 * self.dim.1,
        self.dim.2,
        batch_size,
        self.scale.val.as_ref().as_view().as_ptr(),
        self.bias.val.as_ref().as_view().as_ptr(),
        out_buf.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    in_buf.post(&conn);
    self.scale.val.as_ref().as_view().post(&conn);
    self.bias.val.as_ref().as_view().post(&conn);
    out_buf.post(&conn);
  }

  pub fn _backward<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, out_grad: DeviceMemRef<'a, f32>, mut in_grad: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    in_buf.wait(&conn);
    out_grad.wait(&conn);
    self.scale.val.as_ref().as_view().wait(&conn);
    self.scale.grad.as_mut().as_view().wait(&conn);
    self.bias.grad.as_mut().as_view().wait(&conn);
    in_grad.wait(&conn);
    unsafe { neuralops_cuda_conv2d_scale_bwd(
        in_buf.as_ptr(),
        self.dim.0 * self.dim.1,
        self.dim.2,
        batch_size,
        out_grad.as_ptr(),
        self.scale.val.as_ref().as_view().as_ptr(),
        self.scale.grad.as_mut().as_view_mut().as_mut_ptr(),
        self.bias.grad.as_mut().as_view_mut().as_mut_ptr(),
        in_grad.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    in_buf.post(&conn);
    out_grad.post(&conn);
    self.scale.val.as_ref().as_view().post(&conn);
    self.scale.grad.as_mut().as_view().post(&conn);
    self.bias.grad.as_mut().as_view().post(&conn);
    in_grad.post(&conn);
  }

  pub fn _backward2<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, out_grad: DeviceMemRef<'a, f32>, mut in_grad: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    unimplemented!();
  }

  pub fn _r_forward<'a>(&'a mut self, batch_size: usize, in_val: DeviceMemRef<'a, f32>, in_r_val: DeviceMemRef<'a, f32>, mut out_r_val: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    in_val.wait(&conn);
    in_r_val.wait(&conn);
    self.scale.val.as_ref().as_view().wait(&conn);
    self.scale.r_dir.as_ref().as_view().wait(&conn);
    self.bias.r_dir.as_ref().as_view().wait(&conn);
    out_r_val.wait(&conn);
    unsafe { neuralops_cuda_conv_scale_rfwd(
        in_val.as_ptr(),
        self.dim.0 * self.dim.1,
        self.dim.2,
        batch_size,
        in_r_val.as_ptr(),
        self.scale.val.as_ref().as_view().as_ptr(),
        self.scale.r_dir.as_ref().as_view().as_ptr(),
        self.bias.r_dir.as_ref().as_view().as_ptr(),
        out_r_val.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    in_val.post(&conn);
    in_r_val.post(&conn);
    self.scale.val.as_ref().as_view().post(&conn);
    self.scale.r_dir.as_ref().as_view().post(&conn);
    self.bias.r_dir.as_ref().as_view().post(&conn);
    out_r_val.post(&conn);
  }
}

pub struct DeviceConv2dBatchNormKernel {
  pub dim:      (usize, usize, usize),
  pub epsilon:  f32,
  pub count:    usize,
  pub mean:     DeviceArray1d<f32>,
  pub mean_g:   DeviceArray1d<f32>,
  pub var:      DeviceArray1d<f32>,
  pub var_g:    DeviceArray1d<f32>,
  pub mean_:    Rc<ParamBlock<DeviceArray1d<f32>>>,
  pub var_:     Rc<ParamBlock<DeviceArray1d<f32>>>,
  pub acc_mean: DeviceArray1d<f32>,
  pub acc_var:  DeviceArray1d<f32>,
  pub run_mean: DeviceArray1d<f32>,
  pub run_var:  DeviceArray1d<f32>,
}

impl DeviceConv2dBatchNormKernel {
  pub fn new(dim: (usize, usize, usize), epsilon: f32, stream: DeviceStream) -> Self {
    let chan = dim.2;
    DeviceConv2dBatchNormKernel{
      dim:      dim,
      epsilon:  epsilon,
      count:    0,
      mean:     DeviceArray1d::zeros(chan, stream.conn()),
      mean_g:   DeviceArray1d::zeros(chan, stream.conn()),
      var:      DeviceArray1d::zeros(chan, stream.conn()),
      var_g:    DeviceArray1d::zeros(chan, stream.conn()),
      mean_:    ParamBlock::new(DefaultParamAllocator::new({ let stream = stream.clone(); move || DeviceArray1d::zeros(chan, stream.conn()) })),
      var_:     ParamBlock::new(DefaultParamAllocator::new({ let stream = stream.clone(); move || DeviceArray1d::zeros(chan, stream.conn()) })),
      acc_mean: DeviceArray1d::zeros(chan, stream.conn()),
      acc_var:  DeviceArray1d::zeros(chan, stream.conn()),
      run_mean: DeviceArray1d::zeros(chan, stream.conn()),
      run_var:  DeviceArray1d::zeros(chan, stream.conn()),
    }
  }

  pub fn _forward_inference<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, mut out_buf: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    in_buf.wait(&conn);
    self.run_mean.as_view().wait(&conn);
    self.run_var.as_view().wait(&conn);
    out_buf.wait(&conn);
    unsafe { neuralops_cuda_conv2d_whiten_fwd(
        in_buf.as_ptr(),
        self.dim.0 * self.dim.1,
        self.dim.2,
        batch_size,
        self.run_mean.as_view().as_ptr(),
        self.run_var.as_view().as_ptr(),
        self.epsilon,
        out_buf.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    in_buf.post(&conn);
    self.run_mean.as_view().post(&conn);
    self.run_var.as_view().post(&conn);
    out_buf.post(&conn);
  }

  pub fn _forward_learning<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, mut out_buf: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    self.mean.as_view_mut().set_scalar(0.0, conn.clone());
    self.var.as_view_mut().set_scalar(0.0, conn.clone());
    in_buf.wait(&conn);
    self.mean.as_view().wait(&conn);
    self.var.as_view().wait(&conn);
    out_buf.wait(&conn);
    unsafe { neuralops_cuda_conv2d_mean_fwd(
        in_buf.as_ptr(),
        self.dim.0 * self.dim.1,
        self.dim.2,
        batch_size,
        self.mean.as_view_mut().as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    unsafe { neuralops_cuda_conv2d_var_fwd(
        in_buf.as_ptr(),
        self.dim.0 * self.dim.1,
        self.dim.2,
        batch_size,
        self.mean.as_view().as_ptr(),
        self.var.as_view_mut().as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    unsafe { neuralops_cuda_conv2d_whiten_fwd(
        in_buf.as_ptr(),
        self.dim.0 * self.dim.1,
        self.dim.2,
        batch_size,
        self.mean.as_view().as_ptr(),
        self.var.as_view().as_ptr(),
        self.epsilon,
        out_buf.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    // FIXME: accumulate online mean/variance between batches.
    self.count += batch_size;
    in_buf.post(&conn);
    self.mean.as_view().post(&conn);
    self.var.as_view().post(&conn);
    out_buf.post(&conn);
  }

  pub fn _backward<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, out_grad: DeviceMemRef<'a, f32>, mut in_grad: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    self.mean_g.as_view_mut().set_scalar(0.0, conn.clone());
    self.var_g.as_view_mut().set_scalar(0.0, conn.clone());
    in_buf.wait(&conn);
    out_grad.wait(&conn);
    self.mean.as_view().wait(&conn);
    self.var.as_view().wait(&conn);
    self.mean_g.as_view().wait(&conn);
    self.var_g.as_view().wait(&conn);
    in_grad.wait(&conn);
    unsafe { neuralops_cuda_conv2d_batchnorm_bwd(
        in_buf.as_ptr(),
        self.dim.0 * self.dim.1,
        self.dim.2,
        batch_size,
        out_grad.as_ptr(),
        self.mean.as_view().as_ptr(),
        self.var.as_view().as_ptr(),
        self.epsilon,
        self.mean_g.as_view_mut().as_mut_ptr(),
        self.var_g.as_view_mut().as_mut_ptr(),
        in_grad.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    in_buf.post(&conn);
    out_grad.post(&conn);
    self.mean.as_view().post(&conn);
    self.var.as_view().post(&conn);
    self.mean_g.as_view().post(&conn);
    self.var_g.as_view().post(&conn);
    in_grad.post(&conn);
  }

  pub fn _backward2<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, out_grad: DeviceMemRef<'a, f32>, mut in_grad: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    unimplemented!();
  }

  pub fn _r_forward<'a>(&'a mut self, batch_size: usize, in_val: DeviceMemRef<'a, f32>, in_r_val: DeviceMemRef<'a, f32>, mut out_r_val: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    unimplemented!();
  }

  pub fn _update(&mut self, avg_rate: f32, conn: DeviceConn) {
    self.run_mean.as_view_mut().average(avg_rate, self.mean.as_view(), conn.clone());
    self.run_var.as_view_mut().average(avg_rate, self.var.as_view(), conn.clone());
    //self.run_mean.as_view_mut().average(avg_rate, self.acc_mean.as_view(), conn.clone());
    //self.run_var.as_view_mut().average(avg_rate, self.acc_var.as_view(), conn.clone());
    //self.acc_mean.as_view_mut().set_scalar(0.0, conn.clone());
    //self.acc_var.as_view_mut().set_scalar(0.0, conn.clone());
    self.count = 0;
  }
}

/*pub struct DeviceGemmConv2dOperator<S> {
  cfg:      Conv2dOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf=[f32]>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  hweights: Array4d<f32>,
  hbias:    Array1d<f32>,
  weights:  DeviceArray4d<f32>,
  w_grad:   DeviceArray4d<f32>,
  bias:     DeviceArray1d<f32>,
  b_grad:   DeviceArray1d<f32>,
  col_buf:  DeviceMem<f32>,
  tmp_buf:  DeviceMem<f32>,
  tmp_grad: DeviceMem<f32>,
  act_kern: DeviceActivateKernel,
}

impl<S> DeviceGemmConv2dOperator<S> {
  pub fn new<InOp>(cfg: Conv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceGemmConv2dOperator<S>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf=[f32]> {
    println!("DEBUG: gemm conv2d: {:?}", cfg);
    let (in_w, in_h, in_chan) = cfg.in_dim;
    let (out_w, out_h, out_chan) = cfg.out_dim();
    let col_sz = cfg.kernel_w * cfg.kernel_h * cfg.in_dim.2 * cfg.out_dim().flat_len();
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(DeviceGemmConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap, stream.conn()),
      hweights: Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      hbias:    Array1d::zeros(cfg.out_chan),
      weights:  DeviceArray4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan), stream.conn()),
      w_grad:   DeviceArray4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan), stream.conn()),
      bias:     DeviceArray1d::zeros(cfg.out_chan, stream.conn()),
      b_grad:   DeviceArray1d::zeros(cfg.out_chan, stream.conn()),
      col_buf:  DeviceMem::zeros(col_sz, stream.conn()),
      tmp_buf:  DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      tmp_grad: DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      act_kern: DeviceActivateKernel::new(cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }
}

impl<S> Operator for DeviceGemmConv2dOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> DeviceOperator for DeviceGemmConv2dOperator<S> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> DiffOperator<S> for DeviceGemmConv2dOperator<S> {
  type IoBuf = [f32];

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
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan + self.cfg.out_chan
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
        //let half_range = (6.0 / (self.cfg.in_dim.2 + self.cfg.out_chan) as f64).sqrt();
        let half_range = (3.0 / (self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        //let std = (2.0 / self.cfg.in_dim.2 as f64).sqrt();
        let std = (2.0 / (self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2) as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    self.weights.as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    //self.hbias.as_view_mut().set_constant(0.0);
    //self.bias.as_view_mut().load_sync(self.hbias.as_view(), self.stream.conn());
    self.bias.as_view_mut().set_constant(0.0, self.stream.conn());
  }

  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read_buf(offset, self.hweights.as_mut_slice());
    offset += param_reader.read_buf(offset, self.hbias.as_mut_slice());
    self.weights.as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    self.bias.as_view_mut().load_sync(self.hbias.as_view(), self.stream.conn());
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    self.weights.as_view().store_sync(self.hweights.as_view_mut(), self.stream.conn());
    self.bias.as_view().store_sync(self.hbias.as_view_mut(), self.stream.conn());
    let mut offset = init_offset;
    offset += param_writer.write_buf(offset, self.hweights.as_slice());
    offset += param_writer.write_buf(offset, self.hbias.as_slice());
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut [f32]) -> usize {
    self.w_grad.as_view().store_sync(self.hweights.as_view_mut(), self.stream.conn());
    self.b_grad.as_view().store_sync(self.hbias.as_view_mut(), self.stream.conn());
    let mut offset = init_offset;
    offset += grad_writer.write_buf(offset, self.hweights.as_slice());
    offset += grad_writer.write_buf(offset, self.hbias.as_slice());
    offset - init_offset
  }

  fn _reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0, self.stream.conn());
    self.b_grad.as_view_mut().set_constant(0.0, self.stream.conn());
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    let in_buf = self.in_.buf.borrow();
    in_buf.as_ref().wait(&self.stream.conn());

    let w_in_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2;
    let in_len = self.cfg.in_dim.flat_len();
    let out_len = self.cfg.out_dim().flat_len();
    let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
    for idx in 0 .. batch_size {
      unsafe { neuralops_cuda_caffe_im2col(
          in_buf.as_ref().slice(idx * in_len, (idx+1) * in_len).as_ptr(),
          self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
          self.cfg.kernel_h as _, self.cfg.kernel_w as _,
          self.cfg.pad_h as _, self.cfg.pad_w as _,
          self.cfg.stride_h as _, self.cfg.stride_w as _,
          1, 1,
          self.col_buf.as_mut().as_mut_ptr(),
          self.stream.conn().raw_stream().ptr,
      ) };
      self.tmp_buf.as_mut()
        .slice_mut(idx * out_len, (idx+1) * out_len)
        .reshape_mut((out_space_len, self.cfg.out_chan))
        .matrix_prod(
            1.0,
            self.col_buf.as_ref().reshape((out_space_len, w_in_len)), Transpose::N,
            self.weights.as_view().reshape((w_in_len, self.cfg.out_chan)), Transpose::N,
            0.0,
            self.stream.conn());
    }
    if self.cfg.bias {
      unimplemented!();
    }

    let mut out_buf = self.out.buf.borrow_mut();
    self.act_kern._forward(batch_size, self.tmp_buf.as_ref(), out_buf.as_mut(), self.stream.conn());
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    let out_grad = self.out.grad.as_ref().unwrap().borrow();
    self.act_kern._backward(batch_size, self.tmp_buf.as_ref(), out_grad.as_ref(), self.tmp_grad.as_mut(), self.stream.conn());

    let in_buf = self.in_.buf.borrow();
    in_buf.as_ref().wait(&self.stream.conn());

    let w_in_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2;
    let in_len = self.cfg.in_dim.flat_len();
    let out_len = self.cfg.out_dim().flat_len();
    let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
    for idx in 0 .. batch_size {
      unsafe { neuralops_cuda_caffe_im2col(
          in_buf.as_ref().slice(idx * in_len, (idx+1) * in_len).as_ptr(),
          self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
          self.cfg.kernel_h as _, self.cfg.kernel_w as _,
          self.cfg.pad_h as _, self.cfg.pad_w as _,
          self.cfg.stride_h as _, self.cfg.stride_w as _,
          1, 1,
          self.col_buf.as_mut().as_mut_ptr(),
          self.stream.conn().raw_stream().ptr,
      ) };
      self.w_grad.as_view_mut()
        .reshape_mut((w_in_len, self.cfg.out_chan))
        .matrix_prod(
            1.0,
            self.col_buf.as_ref().reshape((out_space_len, w_in_len)), Transpose::T,
            self.tmp_grad.as_ref().slice(idx * out_len, (idx+1) * out_len).reshape((out_space_len, self.cfg.out_chan)), Transpose::N,
            1.0,
            self.stream.conn(),
        );
    }
    if self.cfg.bias {
      unimplemented!();
    }

    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let mut in_grad = in_grad.borrow_mut();
      in_grad.as_ref().wait(&self.stream.conn());

      // FIXME(20161030)
      let w_in_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2;
      let in_len = self.cfg.in_dim.flat_len();
      let out_len = self.cfg.out_dim().flat_len();
      let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
      in_grad.as_mut().reshape_mut(batch_size * in_len).set_constant(0.0, self.stream.conn());
      for idx in 0 .. batch_size {
        self.col_buf.as_mut()
          .reshape_mut((out_space_len, w_in_len))
          .matrix_prod(
              1.0,
              self.tmp_grad.as_ref().slice(idx * out_len, (idx+1) * out_len).reshape((out_space_len, self.cfg.out_chan)), Transpose::N,
              self.weights.as_view().reshape((w_in_len, self.cfg.out_chan)), Transpose::T,
              0.0,
              self.stream.conn());
        unsafe { neuralops_cuda_caffe_col2im(
            self.col_buf.as_ref().as_ptr(),
            self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
            self.cfg.kernel_h as _, self.cfg.kernel_w as _,
            self.cfg.pad_h as _, self.cfg.pad_w as _,
            self.cfg.stride_h as _, self.cfg.stride_w as _,
            1, 1,
            in_grad.as_mut().slice_mut(idx * in_len, (idx+1) * in_len).as_mut_ptr(),
            self.stream.conn().raw_stream().ptr,
        ) };
      }
    }
  }
}*/

#[derive(Clone, Copy, Debug)]
pub struct CudnnConv2dBackend {
  pub fwd:      usize,
  pub bwd_w:    usize,
  pub bwd_d:    usize,
}

pub struct DeviceConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      Conv2dOperatorConfig,
  //name:     String,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  hweights: Array4d<f32>,
  hbias:    Array1d<f32>,
  /*weights:  DeviceArray4d<f32>,
  w_grad:   DeviceArray4d<f32>,
  bias:     DeviceArray1d<f32>,
  b_grad:   DeviceArray1d<f32>,*/
  weights:  Rc<ParamBlock<DeviceArray4d<f32>>>,
  bias:     Rc<ParamBlock<DeviceArray1d<f32>>>,
  tmp_buf:  DeviceMem<f32>,
  tmp_grad: DeviceMem<f32>,
  tmp:      Rc<VarBlock<DeviceMem<f32>>>,
  scratch:  DeviceMem<u8>,
  add_bias: CudnnAddOp,
  fwd:      CudnnConvFwdOp,
  bwd_w:    CudnnConvBwdFilterOp,
  bwd_d:    CudnnConvBwdDataOp,
  act_kern: DeviceActivateKernel,
}

impl<S, IoBuf: ?Sized> DeviceConv2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: Conv2dOperatorConfig, /*backend: Option<CudnnConv2dBackend>,*/ cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceConv2dOperator<S, IoBuf>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf> {
    //println!("DEBUG: conv2d: {:?}", cfg);
    let (in_w, in_h, in_chan) = cfg.in_dim;
    let (out_w, out_h, out_chan) = cfg.out_dim();
    /*let mut check_out_w: i32 = 0;
    let mut check_out_h: i32 = 0;
    let mut check_out_chan: i32 = 0;
    let mut check_batch_sz: i32 = 0;
    let status = unsafe { cudnnGetConvolution2dForwardOutputDim(
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap().ptr,
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, cfg.batch_sz).unwrap().ptr,
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap().ptr,
        &mut check_batch_sz as *mut _,
        &mut check_out_chan as *mut _,
        &mut check_out_h as *mut _,
        &mut check_out_w as *mut _,
    ) };
    assert!(status.is_ok());
    assert_eq!(out_w, check_out_w as usize);
    assert_eq!(out_h, check_out_h as usize);
    assert_eq!(out_chan, check_out_chan as usize);
    assert_eq!(cfg.batch_sz, check_batch_sz as usize);*/
    let mut workspace_size = 0;
    let fwd = CudnnConvFwdOp::create_fastest(
    /*let fwd = CudnnConvFwdOp::create_algo(
        match backend.unwrap().fwd {
          0 => cudnnConvolutionFwdAlgo_t::ImplicitGemm,
          1 => cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
          2 => cudnnConvolutionFwdAlgo_t::Gemm,
          _ => unimplemented!(),
        },*/
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, fwd.work_size);
    let bwd_w = CudnnConvBwdFilterOp::create_fastest(
    //let bwd_w = CudnnConvBwdFilterOp::create_algo(
    //    cudnnConvolutionBwdFilterAlgo_t::NonDeterministic,
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, 1, 1, out_chan, 1).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, bwd_w.work_size);
    let bwd_d = CudnnConvBwdDataOp::create_fastest(
    //let bwd_d = CudnnConvBwdDataOp::create_algo(
    //    cudnnConvolutionBwdDataAlgo_t::NonDeterministic,
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, bwd_d.work_size);
    //println!("DEBUG: conv2d: workspace size in bytes: {}", workspace_size);
    let add_bias = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, 1, 1, out_chan, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
    );
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(DeviceConv2dOperator{
      cfg:      cfg,
      //name:     String::new(),
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap, stream.clone()),
      hweights: Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      hbias:    Array1d::zeros(cfg.out_chan),
      /*weights:  DeviceArray4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan), stream.conn()),
      w_grad:   DeviceArray4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan), stream.conn()),
      bias:     DeviceArray1d::zeros(cfg.out_chan, stream.conn()),
      b_grad:   DeviceArray1d::zeros(cfg.out_chan, stream.conn()),*/
      weights:  ParamBlock::new(DefaultParamAllocator::new({ let stream = stream.clone(); move || DeviceArray4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan), stream.conn()) })),
      bias:     ParamBlock::new(DefaultParamAllocator::new({ let stream = stream.clone(); move || DeviceArray1d::zeros(cfg.out_chan, stream.conn()) })),
      tmp_buf:  DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      tmp_grad: DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      tmp:      VarBlock::new(DefaultVarAllocator::new({ let stream = stream.clone(); move || DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()) })),
      scratch:  DeviceMem::zeros(workspace_size, stream.conn()),
      add_bias: add_bias,
      fwd:      fwd,
      bwd_w:    bwd_w,
      bwd_d:    bwd_d,
      act_kern: DeviceActivateKernel::new(cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }

  /*pub fn set_name(&mut self, name: &str) {
    self.name = String::from(name);
  }*/
}

impl<S, IoBuf: ?Sized> Operator for DeviceConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> DeviceOperator for DeviceConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for DeviceConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DeviceConv2dOperator<S, IoBuf> {
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

impl<S> DiffOperatorIo<[f32]> for DeviceConv2dOperator<S, [f32]> {
  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    /*offset += param_reader.read_buf(offset, self.hweights.as_mut_slice());
    offset += param_reader.read_buf(offset, self.hbias.as_mut_slice());
    self.weights.as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    self.bias.as_view_mut().load_sync(self.hbias.as_view(), self.stream.conn());*/
    self.weights.val.as_mut().as_view_mut().load_sync(param_reader[offset .. ].reshape((self.cfg.kernel_w, self.cfg.kernel_h, self.cfg.in_dim.2, self.cfg.out_chan)), self.stream.conn());
    offset += self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
    if self.cfg.bias {
      self.bias.val.as_mut().as_view_mut().load_sync(param_reader[offset .. ].reshape(self.cfg.out_chan), self.stream.conn());
      offset += self.cfg.out_chan;
    }
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    /*self.weights.as_view().store_sync(self.hweights.as_view_mut(), self.stream.conn());
    self.bias.as_view().store_sync(self.hbias.as_view_mut(), self.stream.conn());
    offset += param_writer.write_buf(offset, self.hweights.as_slice());
    offset += param_writer.write_buf(offset, self.hbias.as_slice());*/
    self.weights.val.as_ref().as_view().store_sync(param_writer[offset .. ].reshape_mut((self.cfg.kernel_w, self.cfg.kernel_h, self.cfg.in_dim.2, self.cfg.out_chan)), self.stream.conn());
    offset += self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
    if self.cfg.bias {
      self.bias.val.as_ref().as_view().store_sync(param_writer[offset .. ].reshape_mut(self.cfg.out_chan), self.stream.conn());
      offset += self.cfg.out_chan;
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
    self.weights.r_dir.as_mut().as_view_mut().load_sync(dir_reader[offset .. ].reshape((self.cfg.kernel_w, self.cfg.kernel_h, self.cfg.in_dim.2, self.cfg.out_chan)), self.stream.conn());
    offset += self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
    if self.cfg.bias {
      self.bias.r_dir.as_mut().as_view_mut().load_sync(dir_reader[offset .. ].reshape(self.cfg.out_chan), self.stream.conn());
      offset += self.cfg.out_chan;
    }
    offset - init_offset
  }
}

impl<S> DiffOperatorIo<DeviceMem<f32>> for DeviceConv2dOperator<S, DeviceMem<f32>> {
  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut DeviceMem<f32>) -> usize {
    let mut offset = init_offset;
    let w_len = self.weights.val.as_mut().dim().flat_len();
    self.weights.val.as_mut().as_view_mut().reshape_mut(w_len)
      .copy(param_reader.as_ref().slice(offset, offset + w_len).reshape(w_len), self.stream.conn());
    offset += w_len;
    if self.cfg.bias {
      let b_len = self.bias.val.as_mut().dim().flat_len();
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

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for DeviceConv2dOperator<S, IoBuf> {
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
      self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan + self.cfg.out_chan
    } else {
      self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan
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
        //let half_range = (6.0 / (self.cfg.in_dim.2 + self.cfg.out_chan) as f64).sqrt();
        let half_range = (3.0 / (self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        //let std = (2.0 / self.cfg.in_dim.2 as f64).sqrt();
        let std = (2.0 / (self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2) as f64).sqrt();
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
    in_buf.as_ref().wait(&self.stream.conn());
    self.weights.val.as_ref().as_view().wait(&self.stream.conn());
    if self.cfg.bias {
      self.bias.val.as_ref().as_view().wait(&self.stream.conn());
    }
    self.tmp_buf.as_ref().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.fwd.forward(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.weights.val.as_ref().as_view().as_ptr(),
        0.0,
        self.tmp_buf.as_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("cudnn conv fwd failed: {:?}", e); }
    }
    if self.cfg.bias {
      self.add_bias.set_batch_size(batch_size).unwrap();
      unsafe { self.add_bias.forward(
          1.0,
          self.bias.val.as_ref().as_view().as_ptr(),
          1.0,
          self.tmp_buf.as_mut().as_mut_ptr(),
          &*self.stream.conn().cudnn(),
      ).unwrap() };
    }
    in_buf.as_ref().post(&self.stream.conn());
    self.weights.val.as_ref().as_view().post(&self.stream.conn());
    if self.cfg.bias {
      self.bias.val.as_ref().as_view().post(&self.stream.conn());
    }
    self.tmp_buf.as_ref().post(&self.stream.conn());
    self.scratch.as_ref().post(&self.stream.conn());

    let mut out_buf = self.out.buf.borrow_mut();
    self.act_kern._forward(batch_size, self.tmp_buf.as_ref(), out_buf.as_mut(), self.stream.conn());
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    let out_grad = self.out.grad.as_ref().unwrap().borrow();
    self.act_kern._backward(batch_size, self.tmp_buf.as_ref(), out_grad.as_ref(), self.tmp_grad.as_mut(), self.stream.conn());

    let in_buf = self.in_.buf.borrow();
    in_buf.as_ref().wait(&self.stream.conn());
    self.tmp_grad.as_ref().wait(&self.stream.conn());
    self.weights.grad.as_mut().as_view().wait(&self.stream.conn());
    if self.cfg.bias {
      self.bias.grad.as_mut().as_view().wait(&self.stream.conn());
    }
    self.scratch.as_ref().wait(&self.stream.conn());
    self.bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { self.bwd_w.backward_filter(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.tmp_grad.as_ref().as_ptr(),
        1.0,
        self.weights.grad.as_mut().as_view_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    if self.cfg.bias {
      unsafe { self.bwd_w.backward_bias(
          1.0,
          self.tmp_grad.as_ref().as_ptr(),
          1.0,
          self.bias.grad.as_mut().as_view_mut().as_mut_ptr(),
          &*self.stream.conn().cudnn(),
      ).unwrap() };
    }
    in_buf.as_ref().post(&self.stream.conn());
    self.tmp_grad.as_ref().post(&self.stream.conn());
    self.weights.grad.as_mut().as_view().post(&self.stream.conn());
    if self.cfg.bias {
      self.bias.grad.as_mut().as_view().post(&self.stream.conn());
    }
    self.scratch.as_ref().post(&self.stream.conn());

    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let mut in_grad = in_grad.borrow_mut();
      in_grad.as_ref().wait(&self.stream.conn());
      self.tmp_grad.as_ref().wait(&self.stream.conn());
      self.weights.val.as_ref().as_view().wait(&self.stream.conn());
      self.scratch.as_ref().wait(&self.stream.conn());
      self.bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { self.bwd_d.backward_data(
          1.0,
          self.weights.val.as_ref().as_view().as_ptr(),
          self.tmp_grad.as_ref().as_ptr(),
          0.0,
          in_grad.as_mut().as_mut_ptr(),
          self.scratch.as_mut().as_mut_ptr(),
          &*self.stream.conn().cudnn(),
      ).unwrap() };
      in_grad.as_ref().post(&self.stream.conn());
      self.tmp_grad.as_ref().post(&self.stream.conn());
      self.weights.val.as_ref().as_view().post(&self.stream.conn());
      self.scratch.as_ref().post(&self.stream.conn());
    }
  }

  fn _r_forward(&mut self) {
    let batch_size = self.in_.batch_sz.get();

    let in_buf = self.in_.buf.borrow();
    in_buf.as_ref().wait(&self.stream.conn());
    self.in_.data.r_val.as_ref().as_ref().wait(&self.stream.conn());
    self.weights.val.as_ref().as_view().wait(&self.stream.conn());
    self.weights.r_dir.as_ref().as_view().wait(&self.stream.conn());
    if self.cfg.bias {
      self.bias.r_dir.as_ref().as_view().wait(&self.stream.conn());
    }
    self.tmp.r_val.as_mut().as_ref().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.fwd.forward(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.weights.r_dir.as_ref().as_view().as_ptr(),
        0.0,
        self.tmp.r_val.as_mut().as_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("cudnn conv fwd failed: {:?}", e); }
    }
    match unsafe { self.fwd.forward(
        1.0,
        self.in_.data.r_val.as_ref().as_ref().as_ptr(),
        self.weights.val.as_ref().as_view().as_ptr(),
        1.0,
        self.tmp.r_val.as_mut().as_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("cudnn conv fwd failed: {:?}", e); }
    }
    if self.cfg.bias {
      self.add_bias.set_batch_size(batch_size).unwrap();
      unsafe { self.add_bias.forward(
          1.0,
          self.bias.r_dir.as_ref().as_view().as_ptr(),
          1.0,
          self.tmp.r_val.as_mut().as_mut().as_mut_ptr(),
          &*self.stream.conn().cudnn(),
      ).unwrap() };
    }
    in_buf.as_ref().post(&self.stream.conn());
    self.in_.data.r_val.as_ref().as_ref().post(&self.stream.conn());
    self.weights.val.as_ref().as_view().post(&self.stream.conn());
    self.weights.r_dir.as_ref().as_view().post(&self.stream.conn());
    if self.cfg.bias {
      self.bias.r_dir.as_ref().as_view().post(&self.stream.conn());
    }
    self.tmp.r_val.as_mut().as_ref().post(&self.stream.conn());
    self.scratch.as_ref().post(&self.stream.conn());

    self.act_kern._r_forward(batch_size, self.tmp_buf.as_ref(), self.tmp.r_val.as_ref().as_ref(), self.out.data.r_val.as_mut().as_mut(), self.stream.conn());
  }

  fn _r_backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    self.act_kern._r_backward(batch_size, self.tmp_buf.as_ref(), self.tmp.r_val.as_ref().as_ref(), self.out.data.r_grad.as_ref().as_ref(), self.tmp.r_grad.as_mut().as_mut(), self.stream.conn());

    let in_buf = self.in_.buf.borrow();

    in_buf.as_ref().wait(&self.stream.conn());
    self.in_.data.r_val.as_ref().as_ref().wait(&self.stream.conn());
    self.tmp_grad.as_ref().wait(&self.stream.conn());
    self.tmp.r_grad.as_ref().as_ref().wait(&self.stream.conn());
    self.weights.r_grad.as_mut().as_view().wait(&self.stream.conn());
    if self.cfg.bias {
      self.bias.r_grad.as_mut().as_view().wait(&self.stream.conn());
    }
    self.scratch.as_ref().wait(&self.stream.conn());
    self.bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { self.bwd_w.backward_filter(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.tmp.r_grad.as_ref().as_ref().as_ptr(),
        1.0,
        self.weights.r_grad.as_mut().as_view_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    unsafe { self.bwd_w.backward_filter(
        1.0,
        self.in_.data.r_val.as_ref().as_ref().as_ptr(),
        self.tmp_grad.as_ref().as_ptr(),
        1.0,
        self.weights.r_grad.as_mut().as_view_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    if self.cfg.bias {
      unimplemented!();
    }
    in_buf.as_ref().post(&self.stream.conn());
    self.in_.data.r_val.as_ref().as_ref().post(&self.stream.conn());
    self.tmp_grad.as_ref().post(&self.stream.conn());
    self.tmp.r_grad.as_ref().as_ref().post(&self.stream.conn());
    self.weights.r_grad.as_mut().as_view().post(&self.stream.conn());
    if self.cfg.bias {
      self.bias.r_grad.as_mut().as_view().post(&self.stream.conn());
    }
    self.scratch.as_ref().post(&self.stream.conn());

    self.in_.data.r_grad.as_mut().as_ref().wait(&self.stream.conn());
    self.tmp_grad.as_ref().wait(&self.stream.conn());
    self.tmp.r_grad.as_ref().as_ref().wait(&self.stream.conn());
    self.weights.val.as_ref().as_view().wait(&self.stream.conn());
    self.weights.r_dir.as_ref().as_view().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.bwd_d.set_batch_size(batch_size).unwrap();
    unsafe { self.bwd_d.backward_data(
        1.0,
        self.weights.val.as_ref().as_view().as_ptr(),
        self.tmp.r_grad.as_ref().as_ref().as_ptr(),
        0.0,
        self.in_.data.r_grad.as_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    unsafe { self.bwd_d.backward_data(
        1.0,
        self.weights.r_dir.as_ref().as_view().as_ptr(),
        self.tmp_grad.as_ref().as_ptr(),
        1.0,
        self.in_.data.r_grad.as_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    self.in_.data.r_grad.as_mut().as_ref().post(&self.stream.conn());
    self.tmp_grad.as_ref().post(&self.stream.conn());
    self.tmp.r_grad.as_ref().as_ref().post(&self.stream.conn());
    self.weights.val.as_ref().as_view().post(&self.stream.conn());
    self.weights.r_dir.as_ref().as_view().post(&self.stream.conn());
    self.scratch.as_ref().post(&self.stream.conn());
  }

  fn _dump_input(&mut self) -> Vec<u8> {
    let input_sz = self.cfg.batch_sz * self.cfg.in_dim.flat_len() * 4;
    let mut input_data = Vec::with_capacity(input_sz);
    input_data.resize(input_sz, 0);
    {
      let mut input_h = unsafe { from_raw_parts_mut(input_data.as_mut_ptr() as *mut f32, input_data.len() / 4) };
      self.in_.buf.borrow().as_ref().store_sync(&mut input_h, self.stream.conn());
    }
    input_data
  }

  fn _dump_output(&mut self) -> Vec<u8> {
    let output_sz = self.cfg.batch_sz * self.cfg.out_dim().flat_len() * 4;
    let mut output_data = Vec::with_capacity(output_sz);
    output_data.resize(output_sz, 0);
    {
      let mut output_h = unsafe { from_raw_parts_mut(output_data.as_mut_ptr() as *mut f32, output_data.len() / 4) };
      self.tmp_buf.as_ref().store_sync(&mut output_h, self.stream.conn());
    }
    output_data
  }
}

pub struct DeviceBatchNormConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      BatchNormConv2dOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  hweights: Array4d<f32>,
  weights:  Rc<ParamBlock<DeviceArray4d<f32>>>,
  tmp1:     Rc<VarBlock<DeviceMem<f32>>>,
  tmp2:     Rc<VarBlock<DeviceMem<f32>>>,
  tmp3:     Rc<VarBlock<DeviceMem<f32>>>,
  scratch:  DeviceMem<u8>,
  fwd:      CudnnConvFwdOp,
  bwd_w:    CudnnConvBwdFilterOp,
  bwd_d:    CudnnConvBwdDataOp,
  bnorm_k:  DeviceConv2dBatchNormKernel,
  scale_k:  DeviceConv2dScaleKernel,
  act_kern: DeviceActivateKernel,
}

impl<S, IoBuf: ?Sized> DeviceBatchNormConv2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: BatchNormConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceBatchNormConv2dOperator<S, IoBuf>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf> {
    let (in_w, in_h, in_chan) = cfg.in_dim;
    let (out_w, out_h, out_chan) = cfg.out_dim();
    let mut workspace_size = 0;
    let fwd = CudnnConvFwdOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, fwd.work_size);
    let bwd_w = CudnnConvBwdFilterOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, 1, 1, out_chan, 1).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, bwd_w.work_size);
    let bwd_d = CudnnConvBwdDataOp::create_fastest(
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, bwd_d.work_size);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(DeviceBatchNormConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      in_op:    prev_op,
      in_:      in_,
      out:      DeviceOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap, stream.clone()),
      hweights: Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      weights:  ParamBlock::new(DefaultParamAllocator::new({ let stream = stream.clone(); move || DeviceArray4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan), stream.conn()) })),
      tmp1:     VarBlock::new(DefaultVarAllocator::new({ let stream = stream.clone(); move || DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()) })),
      tmp2:     VarBlock::new(DefaultVarAllocator::new({ let stream = stream.clone(); move || DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()) })),
      tmp3:     VarBlock::new(DefaultVarAllocator::new({ let stream = stream.clone(); move || DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()) })),
      scratch:  DeviceMem::zeros(workspace_size, stream.conn()),
      fwd:      fwd,
      bwd_w:    bwd_w,
      bwd_d:    bwd_d,
      bnorm_k:  DeviceConv2dBatchNormKernel::new(cfg.out_dim(), cfg.epsilon, stream.clone()),
      scale_k:  DeviceConv2dScaleKernel::new(cfg.out_dim(), stream.clone()),
      act_kern: DeviceActivateKernel::new(cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for DeviceBatchNormConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> DeviceOperator for DeviceBatchNormConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for DeviceBatchNormConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DeviceBatchNormConv2dOperator<S, IoBuf> {
  default fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }
}

impl<S> DiffOperatorIo<[f32]> for DeviceBatchNormConv2dOperator<S, [f32]> {
  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read_buf(offset, self.hweights.as_mut_slice());
    offset += param_reader.read_buf(offset, self.scale_k.h_scale.as_mut_slice());
    offset += param_reader.read_buf(offset, self.scale_k.h_bias.as_mut_slice());
    self.weights.val.as_mut().as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    self.scale_k.scale.val.as_mut().as_view_mut().load_sync(self.scale_k.h_scale.as_view(), self.stream.conn());
    self.scale_k.bias.val.as_mut().as_view_mut().load_sync(self.scale_k.h_bias.as_view(), self.stream.conn());
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    self.weights.val.as_ref().as_view().store_sync(self.hweights.as_view_mut(), self.stream.conn());
    self.scale_k.scale.val.as_ref().as_view().store_sync(self.scale_k.h_scale.as_view_mut(), self.stream.conn());
    self.scale_k.bias.val.as_ref().as_view().store_sync(self.scale_k.h_bias.as_view_mut(), self.stream.conn());
    let mut offset = init_offset;
    offset += param_writer.write_buf(offset, self.hweights.as_slice());
    offset += param_writer.write_buf(offset, self.scale_k.h_scale.as_slice());
    offset += param_writer.write_buf(offset, self.scale_k.h_bias.as_slice());
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut [f32]) -> usize {
    self.weights.grad.as_ref().as_view().store_sync(self.hweights.as_view_mut(), self.stream.conn());
    self.scale_k.scale.grad.as_ref().as_view().store_sync(self.scale_k.h_scale.as_view_mut(), self.stream.conn());
    self.scale_k.bias.grad.as_ref().as_view().store_sync(self.scale_k.h_bias.as_view_mut(), self.stream.conn());
    let mut offset = init_offset;
    offset += grad_writer.write_buf(offset, self.hweights.as_slice());
    offset += grad_writer.write_buf(offset, self.scale_k.h_scale.as_slice());
    offset += grad_writer.write_buf(offset, self.scale_k.h_bias.as_slice());
    offset - init_offset
  }
}

impl<S> DiffOperatorIo<DeviceMem<f32>> for DeviceBatchNormConv2dOperator<S, DeviceMem<f32>> {
  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut DeviceMem<f32>) -> usize {
    let mut offset = init_offset;
    let w_len = self.weights.val.as_ref().dim().flat_len();
    let s_len = self.scale_k.scale.val.as_ref().dim().flat_len();
    let b_len = self.scale_k.bias.val.as_ref().dim().flat_len();
    self.weights.val.as_mut().as_view_mut().reshape_mut(w_len)
      .copy(param_reader.as_ref().slice(offset, offset + w_len).reshape(w_len), self.stream.conn());
    offset += w_len;
    self.scale_k.scale.val.as_mut().as_view_mut()
      .copy(param_reader.as_ref().slice(offset, offset + s_len).reshape(s_len), self.stream.conn());
    offset += s_len;
    self.scale_k.bias.val.as_mut().as_view_mut()
      .copy(param_reader.as_ref().slice(offset, offset + b_len).reshape(b_len), self.stream.conn());
    offset += b_len;
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut DeviceMem<f32>) -> usize {
    let mut offset = init_offset;
    let w_len = self.weights.val.as_ref().dim().flat_len();
    let s_len = self.scale_k.scale.val.as_ref().dim().flat_len();
    let b_len = self.scale_k.bias.val.as_ref().dim().flat_len();
    param_writer.as_mut().slice_mut(offset, offset + w_len).reshape_mut(w_len)
      .copy(self.weights.val.as_ref().as_view().reshape(w_len), self.stream.conn());
    offset += w_len;
    param_writer.as_mut().slice_mut(offset, offset + s_len).reshape_mut(s_len)
      .copy(self.scale_k.scale.val.as_ref().as_view(), self.stream.conn());
    offset += s_len;
    param_writer.as_mut().slice_mut(offset, offset + b_len).reshape_mut(b_len)
      .copy(self.scale_k.bias.val.as_ref().as_view(), self.stream.conn());
    offset += b_len;
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut DeviceMem<f32>) -> usize {
    let mut offset = init_offset;
    let w_len = self.weights.val.as_ref().dim().flat_len();
    let s_len = self.scale_k.scale.val.as_ref().dim().flat_len();
    let b_len = self.scale_k.bias.val.as_ref().dim().flat_len();
    grad_writer.as_mut().slice_mut(offset, offset + w_len).reshape_mut(w_len)
      .copy(self.weights.grad.as_ref().as_view().reshape(w_len), self.stream.conn());
    offset += w_len;
    grad_writer.as_mut().slice_mut(offset, offset + s_len).reshape_mut(s_len)
      .copy(self.scale_k.scale.grad.as_ref().as_view(), self.stream.conn());
    offset += s_len;
    grad_writer.as_mut().slice_mut(offset, offset + b_len).reshape_mut(b_len)
      .copy(self.scale_k.bias.grad.as_ref().as_view(), self.stream.conn());
    offset += b_len;
    offset - init_offset
  }
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for DeviceBatchNormConv2dOperator<S, IoBuf> {
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
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan + 2 * self.cfg.out_chan
  }

  fn _nondiff_param_sz(&self) -> usize {
    2 * self.cfg.out_chan
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
        //let half_range = (6.0 / (self.cfg.in_dim.2 + self.cfg.out_chan) as f64).sqrt();
        let half_range = (3.0 / (self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        //let std = (2.0 / self.cfg.in_dim.2 as f64).sqrt();
        let std = (2.0 / (self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2) as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    self.weights.val.as_mut().as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    self.bnorm_k.run_mean.as_view_mut().set_constant(0.0, self.stream.conn());
    self.bnorm_k.run_var.as_view_mut().set_constant(1.0, self.stream.conn());
    self.scale_k.scale.val.as_mut().as_view_mut().set_constant(1.0, self.stream.conn());
    self.scale_k.bias.val.as_mut().as_view_mut().set_constant(0.0, self.stream.conn());
  }

  /*fn _accumulate_nondiff(&mut self) {
  }*/

  fn _update_nondiff_param(&mut self, iter: usize) {
    if iter == 0 {
      self.bnorm_k._update(1.0, self.stream.conn());
    } else {
      self.bnorm_k._update(self.cfg.avg_rate, self.stream.conn());
    }
  }

  fn _reset_grad(&mut self) {
    self.weights.grad.as_mut().as_view_mut().set_constant(0.0, self.stream.conn());
    self.scale_k.scale.grad.as_mut().as_view_mut().set_constant(0.0, self.stream.conn());
    self.scale_k.bias.grad.as_mut().as_view_mut().set_constant(0.0, self.stream.conn());
  }

  fn _forward(&mut self, phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    let in_buf = self.in_.buf.borrow();
    in_buf.as_ref().wait(&self.stream.conn());
    self.weights.val.as_ref().as_view().wait(&self.stream.conn());
    self.tmp1.val.as_mut().as_ref().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.fwd.forward(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.weights.val.as_ref().as_view().as_ptr(),
        0.0,
        self.tmp1.val.as_mut().as_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("cudnn conv fwd failed: {:?}", e); }
    }
    in_buf.as_ref().post(&self.stream.conn());
    self.weights.val.as_ref().as_view().post(&self.stream.conn());
    self.tmp1.val.as_mut().as_ref().post(&self.stream.conn());
    self.scratch.as_ref().post(&self.stream.conn());

    let mut out_buf = self.out.buf.borrow_mut();
    match phase {
      OpPhase::Inference => {
        self.bnorm_k._forward_inference(batch_size, self.tmp1.val.as_ref().as_ref(), self.tmp2.val.as_mut().as_mut(), self.stream.conn());
      }
      OpPhase::Learning => {
        self.bnorm_k._forward_learning(batch_size, self.tmp1.val.as_ref().as_ref(), self.tmp2.val.as_mut().as_mut(), self.stream.conn());
      }
    }
    self.scale_k._forward(batch_size, self.tmp2.val.as_ref().as_ref(), self.tmp3.val.as_mut().as_mut(), self.stream.conn());
    self.act_kern._forward(batch_size, self.tmp3.val.as_ref().as_ref(), out_buf.as_mut(), self.stream.conn());
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    let out_grad = self.out.grad.as_ref().unwrap().borrow();
    self.act_kern._backward(batch_size, self.tmp3.val.as_ref().as_ref(), out_grad.as_ref(), self.tmp3.grad.as_mut().as_mut(), self.stream.conn());
    self.scale_k._backward(batch_size, self.tmp2.val.as_ref().as_ref(), self.tmp3.grad.as_ref().as_ref(), self.tmp2.grad.as_mut().as_mut(), self.stream.conn());
    self.bnorm_k._backward(batch_size, self.tmp1.val.as_ref().as_ref(), self.tmp2.grad.as_ref().as_ref(), self.tmp1.grad.as_mut().as_mut(), self.stream.conn());

    let in_buf = self.in_.buf.borrow();
    in_buf.as_ref().wait(&self.stream.conn());
    self.tmp1.grad.as_ref().as_ref().wait(&self.stream.conn());
    self.weights.grad.as_ref().as_view().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { self.bwd_w.backward_filter(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.tmp1.grad.as_ref().as_ref().as_ptr(),
        1.0,
        self.weights.grad.as_mut().as_view_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    in_buf.as_ref().post(&self.stream.conn());
    self.tmp1.grad.as_ref().as_ref().post(&self.stream.conn());
    self.weights.grad.as_ref().as_view().post(&self.stream.conn());
    self.scratch.as_ref().post(&self.stream.conn());

    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let mut in_grad = in_grad.borrow_mut();
      in_grad.as_ref().wait(&self.stream.conn());
      self.tmp1.grad.as_ref().as_ref().wait(&self.stream.conn());
      self.weights.val.as_ref().as_view().wait(&self.stream.conn());
      self.scratch.as_ref().wait(&self.stream.conn());
      self.bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { self.bwd_d.backward_data(
          1.0,
          self.weights.val.as_ref().as_view().as_ptr(),
          self.tmp1.grad.as_ref().as_ref().as_ptr(),
          0.0,
          in_grad.as_mut().as_mut_ptr(),
          self.scratch.as_mut().as_mut_ptr(),
          &*self.stream.conn().cudnn(),
      ).unwrap() };
      in_grad.as_ref().post(&self.stream.conn());
      self.tmp1.grad.as_ref().as_ref().post(&self.stream.conn());
      self.weights.val.as_ref().as_view().post(&self.stream.conn());
      self.scratch.as_ref().post(&self.stream.conn());
    }
  }

  fn _backward2(&mut self) {
    let batch_size = self.out.batch_sz.get();

    // FIXME(20160112): finish implementing bbprop.

    self.weights.val2.as_mut().as_view_mut().copy(self.weights.val.as_ref().as_view(), self.stream.conn());
    self.weights.val2.as_mut().as_view_mut().flatten_mut().square(self.stream.conn());

    //let out_grad = self.out.grad.as_ref().unwrap().borrow();
    let out_grad2 = self.out.grad2(self.stream.conn());
    self.act_kern._backward2(batch_size, self.tmp3.val.as_ref().as_ref(), out_grad2.as_ref(), self.tmp3.grad2.as_mut().as_mut(), self.stream.conn());
    self.scale_k._backward2(batch_size, self.tmp2.val.as_ref().as_ref(), self.tmp3.grad2.as_ref().as_ref(), self.tmp2.grad2.as_mut().as_mut(), self.stream.conn());
    self.bnorm_k._backward2(batch_size, self.tmp1.val.as_ref().as_ref(), self.tmp2.grad2.as_ref().as_ref(), self.tmp1.grad2.as_mut().as_mut(), self.stream.conn());

    let in_buf = self.in_.buf.borrow();
    in_buf.as_ref().wait(&self.stream.conn());
    self.tmp1.grad2.as_ref().as_ref().wait(&self.stream.conn());
    self.weights.grad2.as_mut().as_view().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { self.bwd_w.backward_filter(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.tmp1.grad2.as_ref().as_ref().as_ptr(),
        1.0,
        self.weights.grad2.as_mut().as_view_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    in_buf.as_ref().post(&self.stream.conn());
    self.tmp1.grad2.as_ref().as_ref().post(&self.stream.conn());
    self.weights.grad2.as_mut().as_view().post(&self.stream.conn());
    self.scratch.as_ref().post(&self.stream.conn());

    let mut in_grad2 = self.in_.grad2(self.stream.conn());
    in_grad2.as_ref().wait(&self.stream.conn());
    self.tmp1.grad2.as_ref().as_ref().wait(&self.stream.conn());
    self.weights.val2.as_ref().as_view().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.bwd_d.set_batch_size(batch_size).unwrap();
    unsafe { self.bwd_d.backward_data(
        1.0,
        self.weights.val2.as_ref().as_view().as_ptr(),
        self.tmp1.grad2.as_ref().as_ref().as_ptr(),
        0.0,
        in_grad2.as_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    in_grad2.as_ref().post(&self.stream.conn());
    self.tmp1.grad2.as_ref().as_ref().post(&self.stream.conn());
    self.weights.val2.as_ref().as_view().post(&self.stream.conn());
    self.scratch.as_ref().post(&self.stream.conn());
  }

  fn _r_forward(&mut self) {
    let batch_size = self.in_.batch_sz.get();

    let in_data = self.in_.buf.borrow();
    //let in_r_data = self.in_.r_data.as_ref().unwrap().borrow();
    let in_r_data = self.in_.r_data(self.stream.conn());

    {
      in_data.as_ref().wait(&self.stream.conn());
      in_r_data.as_ref().wait(&self.stream.conn());
      self.weights.val.as_ref().as_view().wait(&self.stream.conn());
      self.weights.r_dir.as_ref().as_view().wait(&self.stream.conn());
      self.tmp1.r_val.as_mut().as_ref().wait(&self.stream.conn());
      self.scratch.as_ref().wait(&self.stream.conn());
      self.fwd.set_batch_size(batch_size).unwrap();
      match unsafe { self.fwd.forward(
          1.0,
          in_r_data.as_ref().as_ptr(),
          self.weights.val.as_ref().as_view().as_ptr(),
          0.0,
          self.tmp1.r_val.as_mut().as_mut().as_mut_ptr(),
          self.scratch.as_mut().as_mut_ptr(),
          &*self.stream.conn().cudnn(),
      ) } {
        Ok(_) => {}
        Err(e) => { panic!("cudnn conv fwd failed: {:?}", e); }
      }
      match unsafe { self.fwd.forward(
          1.0,
          in_data.as_ref().as_ptr(),
          self.weights.r_dir.as_ref().as_view().as_ptr(),
          1.0,
          self.tmp1.r_val.as_mut().as_mut().as_mut_ptr(),
          self.scratch.as_mut().as_mut_ptr(),
          &*self.stream.conn().cudnn(),
      ) } {
        Ok(_) => {}
        Err(e) => { panic!("cudnn conv fwd failed: {:?}", e); }
      }
      in_data.as_ref().post(&self.stream.conn());
      in_r_data.as_ref().post(&self.stream.conn());
      self.weights.val.as_ref().as_view().post(&self.stream.conn());
      self.weights.r_dir.as_ref().as_view().post(&self.stream.conn());
      self.tmp1.r_val.as_mut().as_ref().post(&self.stream.conn());
      self.scratch.as_ref().post(&self.stream.conn());
    }

    //let mut out_r_data = self.out.r_data.as_ref().unwrap().borrow_mut();
    let mut out_r_data = self.out.r_data(self.stream.conn());
    // FIXME(20161216)
    unimplemented!();
    /*self.bnorm_k._r_forward(batch_size, self.t1_buf.as_ref(), self.t2_buf.as_mut(), self.stream.conn());
    self.scale_k._r_forward(batch_size, self.t2_buf.as_ref(), self.t3_buf.as_mut(), self.stream.conn());
    self.act_kern._r_forward(batch_size, self.t3_buf.as_ref(), out_buf.as_mut(), self.stream.conn());*/
  }
}

pub struct DeviceResidualConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      ResidualConv2dOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  join_op:  Rc<RefCell<DeviceAddJoinOperator<S, IoBuf>>>,
  out:      DeviceOutput,
  act_k:    DeviceActivateKernel,
}

impl<S, IoBuf: ?Sized> DeviceResidualConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: ResidualConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceResidualConv2dOperator<S, IoBuf>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf> {
    let split_cfg = SplitOperatorConfig{
      batch_sz: cfg.batch_sz,
      out_arms: 2,
      dim:      cfg.in_dim.flat_len(),
    };
    let conv1_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 3,  kernel_h: 3,
      stride_w: 1,  stride_h: 1,
      pad_w:    1,  pad_h:    1,
      out_chan: cfg.in_dim.2,
      avg_rate: cfg.avg_rate,
      epsilon:  cfg.epsilon,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let conv2_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 3,  kernel_h: 3,
      stride_w: 1,  stride_h: 1,
      pad_w:    1,  pad_h:    1,
      out_chan: cfg.in_dim.2,
      avg_rate: cfg.avg_rate,
      epsilon:  cfg.epsilon,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let join_cfg = JoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      dim:      cfg.in_dim.flat_len(),
    };
    let split_op = DeviceCopySplitOperator::new(split_cfg, cap, prev_op, prev_arm, stream.clone());
    let conv1_op = DeviceBatchNormConv2dOperator::new(conv1_cfg, cap, split_op.clone(), 0, stream.clone());
    let conv2_op = DeviceBatchNormConv2dOperator::new(conv2_cfg, cap, conv1_op, 0, stream.clone());
    let join_op = DeviceAddJoinOperator::new(join_cfg, cap, stream.clone());
    join_op.borrow_mut().append_input(conv2_op, 0);
    join_op.borrow_mut().append_input(split_op, 1);
    Rc::new(RefCell::new(DeviceResidualConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      join_op:  join_op,
      out:      DeviceOutput::new(cfg.batch_sz, cfg.in_dim.flat_len(), cap, stream.clone()),
      act_k:    DeviceActivateKernel::new(cfg.in_dim.flat_len(), cfg.act_kind),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for DeviceResidualConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> DeviceOperator for DeviceResidualConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for DeviceResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DeviceResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for DeviceResidualConv2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.join_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, phase: OpPhase) {
    let join_out = self.join_op.borrow()._output(0);
    let batch_size = join_out.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    let in_buf = join_out.buf.borrow();
    let mut out_buf = self.out.buf.borrow_mut();
    self.act_k._forward(batch_size, in_buf.as_ref(), out_buf.as_mut(), self.stream.conn());
  }

  fn _backward(&mut self) {
    let join_out = self.join_op.borrow()._output(0);
    if let Some(ref join_grad) = join_out.grad.as_ref() {
      let batch_size = self.out.batch_sz.get();
      let in_buf = join_out.buf.borrow();
      let out_grad = self.out.grad.as_ref().unwrap().borrow();
      let mut in_grad = join_grad.borrow_mut();
      self.act_k._backward(batch_size, in_buf.as_ref(), out_grad.as_ref(), in_grad.as_mut(), self.stream.conn());
    }
  }
}

pub struct DeviceProjResidualConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      ProjResidualConv2dOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  join_op:  Rc<RefCell<DeviceAddJoinOperator<S, IoBuf>>>,
  out:      DeviceOutput,
  act_k:    DeviceActivateKernel,
}

impl<S, IoBuf: ?Sized> DeviceProjResidualConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: ProjResidualConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceProjResidualConv2dOperator<S, IoBuf>>> where InOp: 'static + DeviceOperator + DiffOperator<S, IoBuf> {
    let split_cfg = SplitOperatorConfig{
      batch_sz: cfg.batch_sz,
      out_arms: 2,
      dim:      cfg.in_dim.flat_len(),
    };
    let conv1_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 3,  kernel_h: 3,
      stride_w: cfg.stride_w,
      stride_h: cfg.stride_h,
      pad_w:    1,  pad_h:    1,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      epsilon:  cfg.epsilon,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let conv2_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.out_dim(),
      kernel_w: 3,  kernel_h: 3,
      stride_w: 1,  stride_h: 1,
      pad_w:    1,  pad_h:    1,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      epsilon:  cfg.epsilon,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let conv1x1_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 1,  kernel_h: 1,
      stride_w: cfg.stride_w,
      stride_h: cfg.stride_h,
      pad_w:    0,  pad_h:    0,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      epsilon:  cfg.epsilon,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let join_cfg = JoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      dim:      cfg.out_dim().flat_len(),
    };
    let split_op = DeviceCopySplitOperator::new(split_cfg, cap, prev_op, prev_arm, stream.clone());
    let conv1_op = DeviceBatchNormConv2dOperator::new(conv1_cfg, cap, split_op.clone(), 0, stream.clone());
    let conv2_op = DeviceBatchNormConv2dOperator::new(conv2_cfg, cap, conv1_op, 0, stream.clone());
    let conv1x1_op = DeviceBatchNormConv2dOperator::new(conv1x1_cfg, cap, split_op, 1, stream.clone());
    let join_op = DeviceAddJoinOperator::new(join_cfg, cap, stream.clone());
    join_op.borrow_mut().append_input(conv2_op, 0);
    join_op.borrow_mut().append_input(conv1x1_op, 0);
    Rc::new(RefCell::new(DeviceProjResidualConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream.clone(),
      join_op:  join_op,
      out:      DeviceOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap, stream.clone()),
      act_k:    DeviceActivateKernel::new(cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for DeviceProjResidualConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> DeviceOperator for DeviceProjResidualConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for DeviceProjResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DeviceProjResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for DeviceProjResidualConv2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.join_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, phase: OpPhase) {
    let join_out = self.join_op.borrow()._output(0);
    let batch_size = join_out.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    let in_buf = join_out.buf.borrow();
    let mut out_buf = self.out.buf.borrow_mut();
    self.act_k._forward(batch_size, in_buf.as_ref(), out_buf.as_mut(), self.stream.conn());
  }

  fn _backward(&mut self) {
    let join_out = self.join_op.borrow()._output(0);
    if let Some(ref join_grad) = join_out.grad.as_ref() {
      let batch_size = self.out.batch_sz.get();
      let in_buf = join_out.buf.borrow();
      let out_grad = self.out.grad.as_ref().unwrap().borrow();
      let mut in_grad = join_grad.borrow_mut();
      self.act_k._backward(batch_size, in_buf.as_ref(), out_grad.as_ref(), in_grad.as_mut(), self.stream.conn());
    }
  }
}
