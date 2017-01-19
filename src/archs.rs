use prelude::*;

use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use operator::prelude::*;

use std::cell::{RefCell};
use std::rc::{Rc};

const BATCH_NORM_AVG_RATE:  f32 = 0.05;
const BATCH_NORM_EPSILON:   f32 = 1.0e-6;

const INFOGAN_LEAKINESS:    f32 = 0.01;

pub fn build_cifar10_simpleconv_loss<IoBuf: ?Sized + 'static>(batch_sz: usize, stream: DeviceStream) -> Rc<RefCell<DeviceSoftmaxNLLClassLoss<SampleItem, IoBuf>>> {
  let input_cfg = VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 32 * 32 * 3,
    out_dim:    (32, 32, 3),
    //in_dtype:   Dtype::F32,
    in_dtype:   Dtype::U8,
    preprocs:   vec![
      // XXX: the pixel mean is:
      // (1.25306915e2 1.2295039e2 1.1386535e2).
      //VarInputPreproc::ChannelShift{shift: vec![125.0, 123.0, 114.0]},
      VarInputPreproc::Scale{scale: 1.0 / 255.0},
    ],
  };
  let conv1_cfg = Conv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,  kernel_h:   3,
    stride_w:   1,  stride_h:   1,
    pad_w:      1,  pad_h:      1,
    out_chan:   16,
    bias:       true,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let conv2_cfg = Conv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    kernel_w:   3,  kernel_h:   3,
    stride_w:   2,  stride_h:   2,
    pad_w:      1,  pad_h:      1,
    out_chan:   32,
    bias:       true,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let conv3_cfg = Conv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    kernel_w:   3,  kernel_h:   3,
    stride_w:   2,  stride_h:   2,
    pad_w:      1,  pad_h:      1,
    out_chan:   32,
    bias:       true,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let conv4_cfg = Conv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 32),
    kernel_w:   3,  kernel_h:   3,
    stride_w:   2,  stride_h:   2,
    pad_w:      1,  pad_h:      1,
    out_chan:   32,
    bias:       true,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let affine_cfg = AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     512,
    out_dim:    10,
    bias:       false,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  };
  let loss_cfg = ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  };
  let input = DeviceVarInputOperator::new(input_cfg, OpCapability::Backward, stream.clone());
  let conv1 = DeviceConv2dOperator::new(conv1_cfg, OpCapability::Backward, input, 0, stream.clone());
  let conv2 = DeviceConv2dOperator::new(conv2_cfg, OpCapability::Backward, conv1, 0, stream.clone());
  let conv3 = DeviceConv2dOperator::new(conv3_cfg, OpCapability::Backward, conv2, 0, stream.clone());
  let conv4 = DeviceConv2dOperator::new(conv4_cfg, OpCapability::Backward, conv3, 0, stream.clone());
  let affine = DeviceAffineOperator::new(affine_cfg, OpCapability::Backward, conv4, 0, stream.clone());
  let loss = DeviceSoftmaxNLLClassLoss::new(loss_cfg, OpCapability::Backward, affine, 0, stream.clone());
  loss
}

//pub fn build_cifar10_resnet20_loss<S>(batch_sz: usize, stream: DeviceStream) -> Rc<RefCell<DeviceSoftmaxNLLClassLoss<S>>> where S: 'static + SampleDatum<[f32]> + SampleLabel {
pub fn build_cifar10_resnet20_loss<IoBuf: ?Sized + 'static>(batch_sz: usize, augment: bool, stream: DeviceStream) -> Rc<RefCell<DeviceSoftmaxNLLClassLoss<SampleItem, IoBuf>>> {
  let mut preprocs = vec![
      // XXX: the pixel mean is:
      // (1.25306915e2 1.2295039e2 1.1386535e2).
      //VarInputPreproc::ChannelShift{shift: vec![125.0, 123.0, 114.0]},
      VarInputPreproc::Scale{scale: 1.0 / 255.0},
  ];
  if augment {
    preprocs.push(VarInputPreproc::RandomCrop2d{crop_w: 32, crop_h: 32, pad_w: 4, pad_h: 4, phases: vec![OpPhase::Learning]});
    preprocs.push(VarInputPreproc::RandomFlipX{phases: vec![OpPhase::Learning]});
  }
  let input_cfg = VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 32 * 32 * 3,
    out_dim:    (32, 32, 3),
    //in_dtype:   Dtype::F32,
    in_dtype:   Dtype::U8,
    preprocs:   preprocs,
  };
  let conv1_cfg = BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,  kernel_h:   3,
    stride_w:   1,  stride_h:   1,
    pad_w:      1,  pad_h:      1,
    out_chan:   16,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res1_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let proj_res2_cfg = ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    stride_w:   2,  stride_h:   2,
    out_chan:   32,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res2_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let proj_res3_cfg = ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    stride_w:   2,  stride_h:   2,
    out_chan:   64,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res3_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let pool_cfg = Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    pool_w:     8,  pool_h:     8,
    stride_w:   8,  stride_h:   8,
    pad_w:      0,  pad_h:      0,
    kind:       PoolKind::Average,
  };
  let affine_cfg = AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     64,
    out_dim:    10,
    //bias:       true,
    bias:       false,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  };
  let loss_cfg = ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  };
  let input = DeviceVarInputOperator::new(input_cfg, OpCapability::Backward, stream.clone());
  let conv1 = DeviceBatchNormConv2dOperator::new(conv1_cfg, OpCapability::Backward, input, 0, stream.clone());
  let res1_1 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, conv1, 0, stream.clone());
  let res1_2 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_1, 0, stream.clone());
  let res1_3 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_2, 0, stream.clone());
  let res2_1 = DeviceProjResidualConv2dOperator::new(proj_res2_cfg, OpCapability::Backward, res1_3, 0, stream.clone());
  let res2_2 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_1, 0, stream.clone());
  let res2_3 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_2, 0, stream.clone());
  let res3_1 = DeviceProjResidualConv2dOperator::new(proj_res3_cfg, OpCapability::Backward, res2_3, 0, stream.clone());
  let res3_2 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_1, 0, stream.clone());
  let res3_3 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_2, 0, stream.clone());
  let pool = DevicePool2dOperator::new(pool_cfg, OpCapability::Backward, res3_3, 0, stream.clone());
  let affine = DeviceAffineOperator::new(affine_cfg, OpCapability::Backward, pool, 0, stream.clone());
  let loss = DeviceSoftmaxNLLClassLoss::new(loss_cfg, OpCapability::Backward, affine, 0, stream.clone());
  loss
}

//pub fn build_cifar10_resnet56_loss<S>(batch_sz: usize, stream: DeviceStream) -> Rc<RefCell<DeviceSoftmaxNLLClassLoss<S>>> where S: 'static + SampleDatum<[f32]> + SampleLabel {
pub fn build_cifar10_resnet56_loss<IoBuf: ?Sized + 'static>(batch_sz: usize, stream: DeviceStream) -> Rc<RefCell<DeviceSoftmaxNLLClassLoss<SampleItem, IoBuf>>> {
  let input_cfg = VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 32 * 32 * 3,
    out_dim:    (32, 32, 3),
    //in_dtype:   Dtype::F32,
    in_dtype:   Dtype::U8,
    preprocs:   vec![
      // XXX: the pixel mean is:
      // (1.25306915e2 1.2295039e2 1.1386535e2).
      VarInputPreproc::ChannelShift{shift: vec![125.0, 123.0, 114.0]},
      VarInputPreproc::Scale{scale: 1.0 / 256.0},
      VarInputPreproc::RandomCrop2d{crop_w: 32, crop_h: 32, pad_w: 4, pad_h: 4, phases: vec![OpPhase::Learning]},
      VarInputPreproc::RandomFlipX{phases: vec![OpPhase::Learning]},
    ],
  };
  let conv1_cfg = BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,  kernel_h:   3,
    stride_w:   1,  stride_h:   1,
    pad_w:      1,  pad_h:      1,
    out_chan:   16,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res1_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let proj_res2_cfg = ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    stride_w:   2,  stride_h:   2,
    out_chan:   32,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res2_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let proj_res3_cfg = ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    stride_w:   2,  stride_h:   2,
    out_chan:   64,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res3_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let pool_cfg = Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    pool_w:     8,  pool_h:     8,
    stride_w:   8,  stride_h:   8,
    pad_w:      0,  pad_h:      0,
    kind:       PoolKind::Average,
  };
  let affine_cfg = AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     64,
    out_dim:    10,
    bias:       true,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  };
  let loss_cfg = ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  };
  let input = DeviceVarInputOperator::new(input_cfg, OpCapability::Backward, stream.clone());
  let conv1 = DeviceBatchNormConv2dOperator::new(conv1_cfg, OpCapability::Backward, input, 0, stream.clone());
  let res1_1 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, conv1, 0, stream.clone());
  let res1_2 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_1, 0, stream.clone());
  let res1_3 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_2, 0, stream.clone());
  let res1_4 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_3, 0, stream.clone());
  let res1_5 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_4, 0, stream.clone());
  let res1_6 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_5, 0, stream.clone());
  let res1_7 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_6, 0, stream.clone());
  let res1_8 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_7, 0, stream.clone());
  let res1_9 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_8, 0, stream.clone());
  let res2_1 = DeviceProjResidualConv2dOperator::new(proj_res2_cfg, OpCapability::Backward, res1_9, 0, stream.clone());
  let res2_2 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_1, 0, stream.clone());
  let res2_3 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_2, 0, stream.clone());
  let res2_4 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_3, 0, stream.clone());
  let res2_5 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_4, 0, stream.clone());
  let res2_6 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_5, 0, stream.clone());
  let res2_7 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_6, 0, stream.clone());
  let res2_8 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_7, 0, stream.clone());
  let res2_9 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_8, 0, stream.clone());
  let res3_1 = DeviceProjResidualConv2dOperator::new(proj_res3_cfg, OpCapability::Backward, res2_9, 0, stream.clone());
  let res3_2 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_1, 0, stream.clone());
  let res3_3 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_2, 0, stream.clone());
  let res3_4 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_3, 0, stream.clone());
  let res3_5 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_4, 0, stream.clone());
  let res3_6 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_5, 0, stream.clone());
  let res3_7 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_6, 0, stream.clone());
  let res3_8 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_7, 0, stream.clone());
  let res3_9 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_8, 0, stream.clone());
  let pool = DevicePool2dOperator::new(pool_cfg, OpCapability::Backward, res3_9, 0, stream.clone());
  let affine = DeviceAffineOperator::new(affine_cfg, OpCapability::Backward, pool, 0, stream.clone());
  let loss = DeviceSoftmaxNLLClassLoss::new(loss_cfg, OpCapability::Backward, affine, 0, stream.clone());
  loss
}

pub fn build_imagenet_resnet18_loss<IoBuf: ?Sized + 'static>(batch_sz: usize, stream: DeviceStream) -> Rc<RefCell<DeviceSoftmaxNLLClassLoss<SampleItem, IoBuf>>> {
  let input_cfg = VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 16 * 480 * 480 * 3,
    out_dim:    (224, 224, 3),
    //in_dtype:   Dtype::F32,
    in_dtype:   Dtype::U8,
    preprocs:   vec![
      VarInputPreproc::RandomResize2d{hi: 480, lo: 256, phases: vec![OpPhase::Learning]},
      VarInputPreproc::RandomResize2d{hi: 256, lo: 256, phases: vec![OpPhase::Inference]},
      VarInputPreproc::RandomCrop2d{crop_w: 224, crop_h: 224, pad_w: 0, pad_h: 0, phases: vec![OpPhase::Learning]},
      VarInputPreproc::CenterCrop2d{crop_w: 224, crop_h: 224, phases: vec![OpPhase::Inference]},
      VarInputPreproc::RandomFlipX{phases: vec![OpPhase::Learning]},
      //VarInputPreproc::ChannelShift{shift: vec![125.0, 123.0, 114.0]},
      VarInputPreproc::Scale{scale: 1.0 / 256.0},
      //VarInputPreproc::AddPixelwisePCALigtingNoise{},
    ],
  };
  let conv1_cfg = BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (224, 224, 3),
    kernel_w:   7,  kernel_h:   7,
    stride_w:   2,  stride_h:   2,
    pad_w:      3,  pad_h:      3,
    out_chan:   64,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let alt_conv1_cfg = BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (224, 224, 3),
    kernel_w:   7,  kernel_h:   7,
    stride_w:   4,  stride_h:   4,
    pad_w:      3,  pad_h:      3,
    out_chan:   64,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let pool1_cfg = Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (112, 112, 64),
    pool_w:     3,  pool_h:     3,
    stride_w:   2,  stride_h:   2,
    pad_w:      1,  pad_h:      1,
    kind:       PoolKind::Max,
  };
  let res1_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (56, 56, 64),
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let proj_res2_cfg = ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (56, 56, 64),
    stride_w:   2,  stride_h:   2,
    out_chan:   128,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res2_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (28, 28, 128),
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let proj_res3_cfg = ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (28, 28, 128),
    stride_w:   2,  stride_h:   2,
    out_chan:   256,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res3_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (14, 14, 256),
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let proj_res4_cfg = ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (14, 14, 256),
    stride_w:   2,  stride_h:   2,
    out_chan:   512,
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res4_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (7, 7, 512),
    avg_rate:   BATCH_NORM_AVG_RATE,
    epsilon:    BATCH_NORM_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let global_pool_cfg = Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (7, 7, 512),
    pool_w:     7,  pool_h:     7,
    stride_w:   7,  stride_h:   7,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  };
  let affine_cfg = AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     512,
    out_dim:    1000,
    bias:       true,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
    //w_init:     ParamInitKind::Kaiming,
  };
  let loss_cfg = ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    1000,
  };
  let input = DeviceVarInputOperator::new(input_cfg, OpCapability::Backward, stream.clone());
  //let conv1 = DeviceBatchNormConv2dOperator::new(conv1_cfg, OpCapability::Backward, input, 0, stream.clone());
  //let pool1 = DevicePool2dOperator::new(pool1_cfg, OpCapability::Backward, conv1, 0, stream.clone());
  let conv1 = DeviceBatchNormConv2dOperator::new(alt_conv1_cfg, OpCapability::Backward, input, 0, stream.clone());
  let res1_1 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, /*pool1*/ conv1, 0, stream.clone());
  let res1_2 = DeviceResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_1, 0, stream.clone());
  let res2_1 = DeviceProjResidualConv2dOperator::new(proj_res2_cfg, OpCapability::Backward, res1_2, 0, stream.clone());
  let res2_2 = DeviceResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_1, 0, stream.clone());
  let res3_1 = DeviceProjResidualConv2dOperator::new(proj_res3_cfg, OpCapability::Backward, res2_2, 0, stream.clone());
  let res3_2 = DeviceResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_1, 0, stream.clone());
  let res4_1 = DeviceProjResidualConv2dOperator::new(proj_res4_cfg, OpCapability::Backward, res3_2, 0, stream.clone());
  let res4_2 = DeviceResidualConv2dOperator::new(res4_cfg, OpCapability::Backward, res4_1, 0, stream.clone());
  let global_pool = DevicePool2dOperator::new(global_pool_cfg, OpCapability::Backward, res4_2, 0, stream.clone());
  let affine = DeviceAffineOperator::new(affine_cfg, OpCapability::Backward, global_pool, 0, stream.clone());
  let loss = DeviceSoftmaxNLLClassLoss::new(loss_cfg, OpCapability::Backward, affine, 0, stream.clone());
  loss
}
