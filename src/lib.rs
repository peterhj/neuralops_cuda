#![feature(specialization)]

extern crate cuda;
extern crate cuda_dnn;
extern crate densearray;
extern crate devicemem_cuda;
extern crate float;
extern crate iter_utils;
extern crate neuralops;
extern crate neuralops_cuda_kernels;
extern crate operator;
extern crate rng;
//extern crate typemap_alt as typemap;

extern crate libc;
extern crate rand;

pub mod activate;
pub mod affine;
pub mod archs;
pub mod class_loss;
pub mod common;
pub mod conv;
pub mod data;
pub mod deconv;
pub mod input;
pub mod join;
pub mod kernels;
pub mod opt;
pub mod pool;
pub mod prelude;
pub mod regress_loss;
pub mod softmax;
pub mod split;
pub mod util;
