/*use prelude::*;
use activate::{DeviceActivateKernel};
use kernels::*;

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
use std::rc::{Rc};

pub struct DeviceConvTranspose2dOperator<S> {
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

pub struct DeviceBatchNormConvTranspose2dOperator<S> {
  cfg:      BatchNormConv2dOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      DeviceOutput,
  out:      DeviceOutput,
  hweights: Array4d<f32>,
  weights:  DeviceArray4d<f32>,
  w_grad:   DeviceArray4d<f32>,
  t1_buf:   DeviceMem<f32>,
  t1_grad:  DeviceMem<f32>,
  t2_buf:   DeviceMem<f32>,
  t2_grad:  DeviceMem<f32>,
  t3_buf:   DeviceMem<f32>,
  t3_grad:  DeviceMem<f32>,
  scratch:  DeviceMem<u8>,
  fwd:      CudnnConvFwdOp,
  bwd_w:    CudnnConvBwdFilterOp,
  bwd_d:    CudnnConvBwdDataOp,
  bnorm_k:  DeviceConv2dBatchNormKernel,
  scale_k:  DeviceConv2dScaleKernel,
  act_kern: DeviceActivateKernel,
}*/
