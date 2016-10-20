extern crate devicemem_cuda;
extern crate neuralops;
extern crate neuralops_cuda;
extern crate operator;
extern crate rand;
extern crate rng;

use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use neuralops::data::{IndexedDataShard, CyclicDataIter, RandomSampleDataIter, EasyClassLabel};
use neuralops::data::jpeg::{DecodeJpegData};
use neuralops::data::ndarray::{DecodeArray3dData};
use neuralops::data::varraydb::{SharedVarrayDbShard};
use neuralops_cuda::archs::*;
use operator::prelude::*;
use operator::opt::sgd_new::{SgdConfig, SgdWorker};
use operator::opt::shared_sgd_new::{SharedSgdBuilder};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};
use std::path::{PathBuf};
use std::thread::{spawn};

fn main() {
  let batch_sz = 32;
  let minibatch_sz = 32;
  let num_workers = 8;

  let sgd_cfg = SgdConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   minibatch_sz,
    //step_size:      StepSize::Constant(0.1),
    step_size:      StepSize::Decay{init_step: 0.1, step_decay: 0.1, decay_iters: 150000},
    momentum:       Some(GradientMomentum::Nesterov(0.9)),
    checkpoint:     None,
  };
  let builder = SharedSgdBuilder::new(sgd_cfg, num_workers);

  let mut handles = vec![];
  for rank in 0 .. num_workers {
    let builder = builder.clone();
    let handle = spawn(move || {
  let mut train_data =
      DecodeJpegData::new(
      EasyClassLabel::new(
      RandomSampleDataIter::new(
      SharedVarrayDbShard::open(
          PathBuf::from("/scratch/phj/data/ilsvrc2012_v3_shuf/ilsvrc2012_maxscale480_shuf_train_data.varraydb"),
      ))));
  let valid_db = SharedVarrayDbShard::open(
      PathBuf::from("/scratch/phj/data/ilsvrc2012_v3_orig/ilsvrc2012_scale256_orig_valid_data.varraydb"),
  );
  let valid_epoch_sz = valid_db.len();
  let mut valid_data =
      DecodeArray3dData::new(
      EasyClassLabel::new(
      CyclicDataIter::new(
          valid_db
      )));

  let stream = DeviceStream::new(rank);
  let loss = build_imagenet_resnet18_loss(batch_sz, stream);

  //let mut sgd = SgdWorker::new(sgd_cfg, loss);
  let mut sgd = builder.into_worker(rank, loss);

  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  println!("DEBUG: training (CUDA version)...");
  sgd.reset_opt_stats();
  sgd.init_param(&mut rng);
  for iter_nr in 0 .. 600000 {
    sgd.step(&mut train_data);
    if rank == 0 && (iter_nr + 1) % 25 == 0 {
      println!("DEBUG: iter: {} accuracy: {:.3} stats: {:?}", iter_nr + 1, sgd.get_opt_stats().accuracy(), sgd.get_opt_stats());
      sgd.reset_opt_stats();
    }
    if (iter_nr + 1) % 625 == 0 {
      println!("DEBUG: validating...");
      sgd.reset_opt_stats();
      sgd.eval(valid_epoch_sz, &mut valid_data);
      println!("DEBUG: valid: accuracy: {:.3} stats: {:?}", sgd.get_opt_stats().accuracy(), sgd.get_opt_stats());
      sgd.reset_opt_stats();
    }
  }
    });
    handles.push(handle);
  }
  for handle in handles.drain(..) {
    handle.join().unwrap();
  }
}
