extern crate devicemem_cuda;
extern crate neuralops;
extern crate neuralops_cuda;
extern crate operator;
extern crate rand;
extern crate rng;

use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use neuralops::data::{CyclicDataIter, RandomSampleDataIter, EasyClassLabel};
use neuralops::data::cifar::{CifarFlavor, CifarDataShard};
use neuralops::data::jpeg::{DecodeJpegData};
use neuralops::data::varraydb::{SharedVarrayDbShard};
//use neuralops::archs::*;
use neuralops_cuda::archs::*;
use operator::prelude::*;
use operator::opt::sgd_new::{SgdConfig, SgdWorker};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};
use std::path::{PathBuf};

fn main() {
  let batch_sz = 32;
  //let minibatch_sz = 32;
  let minibatch_sz = 256;

  /*let mut train_data =
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
      ));*/

  let mut train_data =
      DecodeJpegData::new(
      EasyClassLabel::new(
      RandomSampleDataIter::new(
      SharedVarrayDbShard::open(
          PathBuf::from("/scratch/phj/data/ilsvrc2012_v3_shuf/ilsvrc2012_maxscale480_shuf_train_data.varraydb"),
      ))));
  let mut valid_data =
      // FIXME: this uses the NdArray serialized format!
      EasyClassLabel::new(
      CyclicDataIter::new(
      SharedVarrayDbShard::open(
          PathBuf::from("/scratch/phj/data/ilsvrc2012_v3_orig/ilsvrc2012_scale256_orig_valid_data.varraydb"),
      )));

  let stream = DeviceStream::new(0);
  //let loss = build_cifar10_resnet20_loss(batch_sz);
  //let loss = build_cifar10_resnet20_loss(batch_sz, stream);
  let loss = build_imagenet_resnet18_loss(batch_sz, stream);

  let sgd_cfg = SgdConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   minibatch_sz,
    //step_size:      StepSize::Constant(0.1),
    step_size:      StepSize::Decay{init_step: 0.1, step_decay: 0.1, decay_iters: 150000},
    momentum:       Some(GradientMomentum::Nesterov(0.9)),
    //checkpoint:     None,
  };
  let mut sgd = SgdWorker::new(sgd_cfg, loss);

  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  println!("DEBUG: training (CUDA version)...");
  sgd.reset_opt_stats();
  sgd.init_param(&mut rng);
  for iter_nr in 0 .. 300000 {
    sgd.step(&mut train_data);
    if (iter_nr + 1) % 1 == 0 {
      println!("DEBUG: iter: {} accuracy: {:.3} stats: {:?}", iter_nr + 1, sgd.get_opt_stats().accuracy(), sgd.get_opt_stats());
      sgd.reset_opt_stats();
    }
    /*if (iter_nr + 1) % 500 == 0 {
      println!("DEBUG: validating...");
      sgd.reset_opt_stats();
      sgd.eval(valid_data.len(), &mut valid_data);
      println!("DEBUG: valid: accuracy: {:.3} stats: {:?}", sgd.get_opt_stats().accuracy(), sgd.get_opt_stats());
      sgd.reset_opt_stats();
    }*/
  }
}
