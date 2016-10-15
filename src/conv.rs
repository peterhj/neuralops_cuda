use prelude::*;
use activate::{DeviceActivateKernel};

use cuda_dnn::v5::{CudnnTensorDesc, CudnnFilterDesc, CudnnConvDesc, CudnnAddOp, CudnnConvFwdOp, CudnnConvBwdFilterOp, CudnnConvBwdDataOp};
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
//use std::marker::{PhantomData};
use std::rc::{Rc};

pub struct DeviceConv2dScaleKernel {
  pub dim:      (usize, usize, usize),
  pub scale:    DeviceArray1d<f32>,
  pub scale_g:  DeviceArray1d<f32>,
  pub bias:     DeviceArray1d<f32>,
  pub bias_g:   DeviceArray1d<f32>,
}

impl DeviceConv2dScaleKernel {
  pub fn new(dim: (usize, usize, usize), conn: DeviceConn) -> Self {
    unimplemented!();
  }

  pub fn _forward<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, mut out_buf: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    // FIXME: wait.
    unsafe { neuralops_cuda_conv2d_scale_fwd(
        in_buf.as_ptr(),
        dim.0 * dim.1,
        dim.2,
        batch_size,
        self.scale.as_view().as_ptr(),
        self.bias.as_view().as_ptr(),
        out_buf.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    // FIXME: post.
  }

  pub fn _backward<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, out_grad: DeviceMemRef<'a, f32>, mut in_grad: DeviceMemRef<'a, f32>, conn: DeviceConn) {
    // FIXME: wait.
    unsafe { neuralops_cuda_conv2d_scale_bwd(
        in_buf.as_ptr(),
        dim.0 * dim.1,
        dim.2,
        batch_size,
        out_grad.as_ptr(),
        self.scale.as_view().as_ptr(),
        self.scale_g.as_view_mut().as_mut_ptr(),
        self.bias_g.as_view_mut().as_mut_ptr(),
        in_grad.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    // FIXME: post.
  }
}

pub struct DeviceConv2dBatchNormKernel {
  pub dim:      (usize, usize, usize),
  pub epsilon:  f32,
  pub count:    usize,
  pub mean:     DeviceArray1d<f32>,
  pub mean_g:   DeviceArray1d<f32>,
  pub mean_acc: DeviceArray1d<f32>,
  pub run_mean: DeviceArray1d<f32>,
  pub var:      DeviceArray1d<f32>,
  pub var_g:    DeviceArray1d<f32>,
  pub var_acc:  DeviceArray1d<f32>,
  pub run_var:  DeviceArray1d<f32>,
}

impl DeviceConv2dBatchNormKernel {
  pub fn new(dim: (usize, usize, usize), epsilon: f32, conn: DeviceConn) -> Self {
    unimplemented!();
  }

  pub fn _forward_inference<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, mut out_buf: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    // FIXME: wait.
    unsafe { neuralops_cuda_conv2d_whiten_fwd(
        in_buf.as_ptr(),
        dim.0 * dim.1,
        dim.2,
        batch_size,
        self.run_mean.as_view().as_ptr(),
        self.run_var.as_view().as_ptr(),
        out_buf.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    // FIXME: post.
  }

  pub fn _forward_learning<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, mut out_buf: DeviceMemRefMut<'a, f32>, conn: DeviceConn) {
    // FIXME: wait.
    self.mean.as_view_mut().set_scalar(0.0, conn.clone());
    self.var.as_view_mut().set_scalar(0.0, conn.clone());
    unsafe { neuralops_cuda_conv2d_mean_fwd(
        in_buf.as_ptr(),
        dim.0 * dim.1,
        dim.2,
        batch_size,
        self.mean.as_view_mut().as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    unsafe { neuralops_cuda_conv2d_var_fwd(
        in_buf.as_ptr(),
        dim.0 * dim.1,
        dim.2,
        batch_size,
        self.mean.as_view().as_ptr(),
        self.var.as_view_mut().as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    unsafe { neuralops_cuda_conv2d_whiten_fwd(
        in_buf.as_ptr(),
        dim.0 * dim.1,
        dim.2,
        batch_size,
        self.mean.as_view().as_ptr(),
        self.var.as_view().as_ptr(),
        self.epsilon,
        out_buf.as_mut_ptr(),
        conn.raw_stream().ptr,
    ) };
    // FIXME: accumulate online mean/variance between batches.
    self.count += 1;
    // FIXME: post.
  }

  pub fn _backward<'a>(&'a mut self, batch_size: usize, in_buf: DeviceMemRef<'a, f32>, out_grad: DeviceMemRef<'a, f32>, mut in_grad: DeviceMemRef<'a, f32>, conn: DeviceConn) {
    // FIXME: wait.
    unsafe { neuralops_cuda_conv2d_batchnorm_bwd(
        in_buf.as_ptr(),
        dim.0 * dim.1,
        dim.2,
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
    // FIXME: post.
  }

  pub fn _update(&mut self, avg_rate: f32, conn: DeviceConn) {
    self.run_mean.as_view_mut().average(avg_rate, self.mean.as_view(), conn.clone());
    self.run_var.as_view_mut().average(avg_rate, self.var.as_view(), conn.clone());
    //self.run_mean.as_view_mut().average(avg_rate, self.mean_acc.as_view(), conn.clone());
    //self.run_var.as_view_mut().average(avg_rate, self.var_acc.as_view(), conn.clone());
    //self.mean_acc.as_view_mut().set_scalar(0.0, conn.clone());
    //self.var_acc.as_view_mut().set_scalar(0.0, conn.clone());
    self.count = 0;
  }
}

pub struct DeviceConv2dOperator<S> {
  cfg:      Conv2dOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  hweights: Array4d<f32>,
  hbias:    Array1d<f32>,
  weights:  DeviceArray4d<f32>,
  w_grad:   DeviceArray4d<f32>,
  bias:     DeviceArray1d<f32>,
  b_grad:   DeviceArray1d<f32>,
  tmp_buf:  DeviceMem<f32>,
  tmp_grad: DeviceMem<f32>,
  scratch:  DeviceMem<u8>,
  add_bias: CudnnAddOp,
  fwd:      CudnnConvFwdOp,
  bwd_w:    CudnnConvBwdFilterOp,
  bwd_d:    CudnnConvBwdDataOp,
  act_kern: DeviceActivateKernel,
}

impl<S> DeviceConv2dOperator<S> {
  pub fn new<InOp>(cfg: Conv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceConv2dOperator<S>>> where InOp: 'static + DeviceOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let (in_w, in_h, in_chan) = cfg.in_dim;
    let (out_w, out_h, out_chan) = cfg.out_dim();
    let mut workspace_size = 0;
    let add_bias = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_chan, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
    );
    let fwd = CudnnConvFwdOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, fwd.work_size);
    let bwd_w = CudnnConvBwdFilterOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_chan, 1).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, bwd_w.work_size);
    let bwd_d = CudnnConvBwdDataOp::create_fastest(
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, bwd_d.work_size);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(DeviceConv2dOperator{
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
      tmp_buf:  DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      tmp_grad: DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      scratch:  DeviceMem::zeros(workspace_size, stream.conn()),
      add_bias: add_bias,
      fwd:      fwd,
      bwd_w:    bwd_w,
      bwd_d:    bwd_d,
      act_kern: DeviceActivateKernel::new(cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }
}

impl<S> Operator for DeviceConv2dOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> DeviceOperator for DeviceConv2dOperator<S> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for DeviceConv2dOperator<S> {
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
        let half_range = (6.0 / (self.cfg.in_dim.2 + self.cfg.out_chan) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        let std = (2.0 / self.cfg.in_dim.2 as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    self.hbias.as_view_mut().set_constant(0.0);
    self.weights.as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    self.bias.as_view_mut().load_sync(self.hbias.as_view(), self.stream.conn());
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
    self.weights.as_view().wait(&self.stream.conn());
    self.bias.as_view().wait(&self.stream.conn());
    self.tmp_buf.as_ref().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.fwd.forward(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.weights.as_view().as_ptr(),
        0.0,
        self.tmp_buf.as_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("cudnn conv fwd failed: {:?}", e); }
    }
    self.add_bias.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias.forward(
        1.0,
        self.bias.as_view().as_ptr(),
        1.0,
        self.tmp_buf.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    in_buf.as_ref().post(&self.stream.conn());
    self.weights.as_view().post(&self.stream.conn());
    self.bias.as_view().post(&self.stream.conn());
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
    self.w_grad.as_view().wait(&self.stream.conn());
    self.b_grad.as_view().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { self.bwd_w.backward_filter(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.tmp_grad.as_ref().as_ptr(),
        1.0,
        self.w_grad.as_view_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    unsafe { self.bwd_w.backward_bias(
        1.0,
        self.tmp_grad.as_ref().as_ptr(),
        1.0,
        self.b_grad.as_view_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    in_buf.as_ref().post(&self.stream.conn());
    self.tmp_grad.as_ref().post(&self.stream.conn());
    self.w_grad.as_view().post(&self.stream.conn());
    self.b_grad.as_view().post(&self.stream.conn());
    self.scratch.as_ref().post(&self.stream.conn());

    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let mut in_grad = in_grad.borrow_mut();
      in_grad.as_ref().wait(&self.stream.conn());
      self.tmp_grad.as_ref().wait(&self.stream.conn());
      self.weights.as_view().wait(&self.stream.conn());
      self.scratch.as_ref().wait(&self.stream.conn());
      self.bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { self.bwd_d.backward_data(
          1.0,
          self.weights.as_view().as_ptr(),
          self.tmp_grad.as_ref().as_ptr(),
          0.0,
          in_grad.as_mut().as_mut_ptr(),
          self.scratch.as_mut().as_mut_ptr(),
          &*self.stream.conn().cudnn(),
      ).unwrap() };
      in_grad.as_ref().post(&self.stream.conn());
      self.tmp_grad.as_ref().post(&self.stream.conn());
      self.weights.as_view().post(&self.stream.conn());
      self.scratch.as_ref().post(&self.stream.conn());
    }
  }
}

pub struct DeviceBatchNormConv2dOperator<S> {
  cfg:      BatchNormConv2dOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  hweights: Array4d<f32>,
  //hbias:    Array1d<f32>,
  weights:  DeviceArray4d<f32>,
  w_grad:   DeviceArray4d<f32>,
  //bias:     DeviceArray1d<f32>,
  //b_grad:   DeviceArray1d<f32>,
  t1_buf:   DeviceMem<f32>,
  t1_grad:  DeviceMem<f32>,
  t2_buf:   DeviceMem<f32>,
  t2_grad:  DeviceMem<f32>,
  t3_buf:   DeviceMem<f32>,
  t3_grad:  DeviceMem<f32>,
  scratch:  DeviceMem<u8>,
  //add_bias: CudnnAddOp,
  fwd:      CudnnConvFwdOp,
  bwd_w:    CudnnConvBwdFilterOp,
  bwd_d:    CudnnConvBwdDataOp,
  bnorm_k:  DeviceConv2dBatchNormKernel,
  scale_k:  DeviceConv2dScaleKernel,
  act_kern: DeviceActivateKernel,
}

impl<S> DeviceBatchNormConv2dOperator<S> {
  pub fn new<InOp>(cfg: BatchNormConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize, stream: DeviceStream) -> Rc<RefCell<DeviceBatchNormConv2dOperator<S>>> where InOp: 'static + DeviceOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let (in_w, in_h, in_chan) = cfg.in_dim;
    let (out_w, out_h, out_chan) = cfg.out_dim();
    let mut workspace_size = 0;
    /*let add_bias = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_chan, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
    );*/
    let fwd = CudnnConvFwdOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, fwd.work_size);
    let bwd_w = CudnnConvBwdFilterOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_chan, 1).unwrap(),
        &*stream.conn().cudnn(),
    ).unwrap();
    workspace_size = max(workspace_size, bwd_w.work_size);
    let bwd_d = CudnnConvBwdDataOp::create_fastest(
        CudnnFilterDesc::<f32>::create_4d(cfg.kernel_w, cfg.kernel_h, in_chan, out_chan).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_w, out_h, out_chan, cfg.batch_sz).unwrap(),
        CudnnConvDesc::create_2d(cfg.stride_w, cfg.stride_h, cfg.pad_w, cfg.pad_h).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_w, in_h, in_chan, cfg.batch_sz).unwrap(),
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
      out:      DeviceOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap, stream.conn()),
      hweights: Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      //hbias:    Array1d::zeros(cfg.out_chan),
      weights:  DeviceArray4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan), stream.conn()),
      w_grad:   DeviceArray4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan), stream.conn()),
      //bias:     DeviceArray1d::zeros(cfg.out_chan, stream.conn()),
      //b_grad:   DeviceArray1d::zeros(cfg.out_chan, stream.conn()),
      t1_buf:   DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      t1_grad:  DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      t2_buf:   DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      t2_grad:  DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      t3_buf:   DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      t3_grad:  DeviceMem::zeros(cfg.batch_sz * cfg.out_dim().flat_len(), stream.conn()),
      scratch:  DeviceMem::zeros(workspace_size, stream.conn()),
      //add_bias: add_bias,
      fwd:      fwd,
      bwd_w:    bwd_w,
      bwd_d:    bwd_d,
      bnorm_k:  DeviceConv2dBatchNormKernel::new(),
      scale_k:  DeviceConv2dScaleKernel::new(),
      act_kern: DeviceActivateKernel::new(cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }
}

impl<S> Operator for DeviceBatchNormConv2dOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> DeviceOperator for DeviceBatchNormConv2dOperator<S> {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for DeviceBatchNormConv2dOperator<S> {
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
        let half_range = (6.0 / (self.cfg.in_dim.2 + self.cfg.out_chan) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        let std = (2.0 / self.cfg.in_dim.2 as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.hweights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    //self.hbias.as_view_mut().set_constant(0.0);
    self.weights.as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    //self.bias.as_view_mut().load_sync(self.hbias.as_view(), self.stream.conn());
  }

  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read_buf(offset, self.hweights.as_mut_slice());
    //offset += param_reader.read_buf(offset, self.hbias.as_mut_slice());
    self.weights.as_view_mut().load_sync(self.hweights.as_view(), self.stream.conn());
    //self.bias.as_view_mut().load_sync(self.hbias.as_view(), self.stream.conn());
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    self.weights.as_view().store_sync(self.hweights.as_view_mut(), self.stream.conn());
    //self.bias.as_view().store_sync(self.hbias.as_view_mut(), self.stream.conn());
    let mut offset = init_offset;
    offset += param_writer.write_buf(offset, self.hweights.as_slice());
    //offset += param_writer.write_buf(offset, self.hbias.as_slice());
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut [f32]) -> usize {
    self.w_grad.as_view().store_sync(self.hweights.as_view_mut(), self.stream.conn());
    //self.b_grad.as_view().store_sync(self.hbias.as_view_mut(), self.stream.conn());
    let mut offset = init_offset;
    offset += grad_writer.write_buf(offset, self.hweights.as_slice());
    //offset += grad_writer.write_buf(offset, self.hbias.as_slice());
    offset - init_offset
  }

  fn _reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0, self.stream.conn());
    //self.b_grad.as_view_mut().set_constant(0.0, self.stream.conn());
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    let in_buf = self.in_.buf.borrow();
    in_buf.as_ref().wait(&self.stream.conn());
    self.weights.as_view().wait(&self.stream.conn());
    //self.bias.as_view().wait(&self.stream.conn());
    self.t1_buf.as_ref().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.fwd.forward(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.weights.as_view().as_ptr(),
        0.0,
        self.t1_buf.as_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("cudnn conv fwd failed: {:?}", e); }
    }
    /*self.add_bias.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias.forward(
        1.0,
        self.bias.as_view().as_ptr(),
        1.0,
        self.t1_buf.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };*/
    in_buf.as_ref().post(&self.stream.conn());
    self.weights.as_view().post(&self.stream.conn());
    //self.bias.as_view().post(&self.stream.conn());
    self.t1_buf.as_ref().post(&self.stream.conn());
    self.scratch.as_ref().post(&self.stream.conn());

    let mut out_buf = self.out.buf.borrow_mut();
    self.act_kern._forward(batch_size, self.t1_buf.as_ref(), out_buf.as_mut(), self.stream.conn());
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    let out_grad = self.out.grad.as_ref().unwrap().borrow();
    self.act_kern._backward(batch_size, self.t1_buf.as_ref(), out_grad.as_ref(), self.t1_grad.as_mut(), self.stream.conn());

    let in_buf = self.in_.buf.borrow();
    in_buf.as_ref().wait(&self.stream.conn());
    self.t1_grad.as_ref().wait(&self.stream.conn());
    self.w_grad.as_view().wait(&self.stream.conn());
    //self.b_grad.as_view().wait(&self.stream.conn());
    self.scratch.as_ref().wait(&self.stream.conn());
    self.bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { self.bwd_w.backward_filter(
        1.0,
        in_buf.as_ref().as_ptr(),
        self.t1_grad.as_ref().as_ptr(),
        1.0,
        self.w_grad.as_view_mut().as_mut_ptr(),
        self.scratch.as_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };
    /*unsafe { self.bwd_w.backward_bias(
        1.0,
        self.t1_grad.as_ref().as_ptr(),
        1.0,
        self.b_grad.as_view_mut().as_mut_ptr(),
        &*self.stream.conn().cudnn(),
    ).unwrap() };*/
    in_buf.as_ref().post(&self.stream.conn());
    self.t1_grad.as_ref().post(&self.stream.conn());
    self.w_grad.as_view().post(&self.stream.conn());
    //self.b_grad.as_view().post(&self.stream.conn());
    self.scratch.as_ref().post(&self.stream.conn());

    if let Some(ref in_grad) = self.in_.grad.as_ref() {
      let mut in_grad = in_grad.borrow_mut();
      in_grad.as_ref().wait(&self.stream.conn());
      self.t1_grad.as_ref().wait(&self.stream.conn());
      self.weights.as_view().wait(&self.stream.conn());
      self.scratch.as_ref().wait(&self.stream.conn());
      self.bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { self.bwd_d.backward_data(
          1.0,
          self.weights.as_view().as_ptr(),
          self.t1_grad.as_ref().as_ptr(),
          0.0,
          in_grad.as_mut().as_mut_ptr(),
          self.scratch.as_mut().as_mut_ptr(),
          &*self.stream.conn().cudnn(),
      ).unwrap() };
      in_grad.as_ref().post(&self.stream.conn());
      self.t1_grad.as_ref().post(&self.stream.conn());
      self.weights.as_view().post(&self.stream.conn());
      self.scratch.as_ref().post(&self.stream.conn());
    }
  }
}
