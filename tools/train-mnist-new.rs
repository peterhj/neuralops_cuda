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
//use neuralops::affine::{NewAffineOperator};
//use neuralops::input::{VarInputPreproc, NewVarInputOperator};
//use neuralops::class_loss::{SoftmaxNLLClassLoss};
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
  let affine1 = DeviceAffineOperator::new(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     784,
    out_dim:    50,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }, OpCapability::Backward, input, 0, stream.clone());
  let affine2 = DeviceAffineOperator::new(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     50,
    out_dim:    10,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }, OpCapability::Backward, affine1, 0, stream.clone());
  let loss = DeviceSoftmaxNLLClassLoss::new(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }, OpCapability::Backward, affine2, 0, stream.clone());

  let sgd_cfg = SgdConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    step_size:      StepSize::Constant(0.1),
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
