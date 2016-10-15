extern crate devicemem_cuda;
extern crate neuralops;
extern crate neuralops_cuda;
extern crate operator;
extern crate rand;
extern crate rng;

use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use neuralops::data::{CyclicDataIter, RandomSampleDataIter};
use neuralops::data::cifar::{CifarFlavor, CifarDataShard};
//use neuralops::archs::*;
use neuralops_cuda::archs::*;
use operator::prelude::*;
use operator::opt::sgd_new::{SgdConfig, SgdWorker};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};
use std::path::{PathBuf};

fn main() {
  let batch_sz = 128;

  let mut train_data =
      RandomSampleDataIter::new(
      CifarDataShard::new(
          CifarFlavor::Cifar10,
          PathBuf::from("datasets/cifar10/train.bin"),
      ));
  let mut valid_data =
      CyclicDataIter::new(
      CifarDataShard::new(
          CifarFlavor::Cifar10,
          PathBuf::from("datasets/cifar10/test.bin"),
      ));

  let stream = DeviceStream::new(0);
  //let loss = build_cifar10_resnet20_loss(batch_sz);
  let loss = build_cifar10_resnet20_loss(batch_sz, stream);

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
  for iter_nr in 0 .. 100000 {
    sgd.step(&mut train_data);
    if (iter_nr + 1) % 10 == 0 {
      println!("DEBUG: iter: {} accuracy: {:.3} stats: {:?}", iter_nr + 1, sgd.get_opt_stats().accuracy(), sgd.get_opt_stats());
      sgd.reset_opt_stats();
    }
    if (iter_nr + 1) % 500 == 0 {
      println!("DEBUG: validating...");
      sgd.reset_opt_stats();
      sgd.eval(valid_data.len(), &mut valid_data);
      println!("DEBUG: valid: accuracy: {:.3} stats: {:?}", sgd.get_opt_stats().accuracy(), sgd.get_opt_stats());
      sgd.reset_opt_stats();
    }
  }
}
