extern crate devicemem_cuda;
extern crate neuralops;
extern crate neuralops_cuda;
extern crate operator;
extern crate rand;
extern crate rng;

use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use neuralops::data::{CyclicDataIter, SubsampleDataIter};
use neuralops::data::mnist::{MnistDataShard};
use neuralops_cuda::prelude::*;
use operator::prelude::*;
use operator::opt::sgd_new::{SgdConfig, SgdWorker};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};
use std::path::{PathBuf};

fn main() {
  let batch_sz = 32;

  let stream = DeviceStream::new(0);

  let mut train_data =
      SubsampleDataIter::new(
      batch_sz,
      MnistDataShard::new(
          PathBuf::from("datasets/mnist/train-images-idx3-ubyte"),
          PathBuf::from("datasets/mnist/train-labels-idx1-ubyte"),
      ));
  let mut valid_data =
      CyclicDataIter::new(
      MnistDataShard::new(
          PathBuf::from("datasets/mnist/t10k-images-idx3-ubyte"),
          PathBuf::from("datasets/mnist/t10k-labels-idx1-ubyte"),
      ));

  let input = DeviceVarInputOperator::new(VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 784,
    out_dim:    (28, 28, 1),
    preprocs:   vec![
      VarInputPreproc::Scale{scale: 1.0 / 255.0},
    ],
  }, OpCapability::Backward, stream.clone());
  /*let affine1 = DeviceAffineOperator::new(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     784,
    out_dim:    50,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }, OpCapability::Backward, input, 0, stream.clone());*/
  //let conv1 = DeviceConv2dOperator::new(Conv2dOperatorConfig{
  let conv1 = DeviceBatchNormConv2dOperator::new(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (28, 28, 1),
    kernel_w:   5,  kernel_h:   5,
    stride_w:   2,  stride_h:   2,
    pad_w:      2,  pad_h:      2,
    out_chan:   16,
    avg_rate:   0.05,
    epsilon:    1.0e-6,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }, OpCapability::Backward, input, 0, stream.clone());
  //let conv2 = DeviceConv2dOperator::new(Conv2dOperatorConfig{
  let conv2 = DeviceBatchNormConv2dOperator::new(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (14, 14, 16),
    kernel_w:   3,  kernel_h:   3,
    stride_w:   1,  stride_h:   1,
    pad_w:      1,  pad_h:      1,
    out_chan:   32,
    avg_rate:   0.05,
    epsilon:    1.0e-6,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }, OpCapability::Backward, conv1, 0, stream.clone());
  let affine2 = DeviceAffineOperator::new(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     14 * 14 * 32,
    out_dim:    10,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }, OpCapability::Backward, conv2, 0, stream.clone());
  let loss = DeviceSoftmaxNLLClassLoss::new(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }, OpCapability::Backward, affine2, 0, stream.clone());

  let sgd_cfg = SgdConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    step_size:      StepSize::Constant(0.01),
    momentum:       Some(GradientMomentum::Nesterov(0.9)),
  };
  let mut sgd = SgdWorker::new(sgd_cfg, loss);

  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  println!("DEBUG: training (CUDA version)...");
  sgd.reset_opt_stats();
  sgd.init_param(&mut rng);
  for iter_nr in 0 .. 1000 {
    sgd.step(&mut train_data);
    if (iter_nr + 1) % 20 == 0 {
      println!("DEBUG: iter: {} stats: {:?}", iter_nr + 1, sgd.get_opt_stats());
      sgd.reset_opt_stats();
    }
  }
  println!("DEBUG: validation...");
  sgd.reset_opt_stats();
  sgd.eval(valid_data.len(), &mut valid_data);
  println!("DEBUG: valid stats: {:?}", sgd.get_opt_stats());
}
