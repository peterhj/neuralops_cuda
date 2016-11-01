use prelude::*;
use kernels::*;

use densearray::{ArrayIndex, ReshapeMut};
use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use operator::prelude::*;
use rng::{RngState};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use std::cell::{RefCell};
use std::marker::{PhantomData};
use std::rc::{Rc};

pub struct DeviceVarInputOperator<S> {
  cfg:      VarInputOperatorConfig,
  node:     OperatorNode,
  stream:   DeviceStream,
  out:      DeviceOutput,
  rng:      Xorshiftplus128Rng,
  r_state:  Vec<u64>,
  in_dims:  Vec<(usize, usize, usize)>,
  tmp_dims: Vec<(usize, usize, usize)>,
  h_buf:    Vec<f32>,
  tmp_buf:  DeviceMem<f32>,
  _marker:  PhantomData<S>,
}

impl<S> DeviceVarInputOperator<S> /*where S: SampleDatum<[f32]>*/ {
  pub fn new(cfg: VarInputOperatorConfig, cap: OpCapability, stream: DeviceStream) -> Rc<RefCell<DeviceVarInputOperator<S>>> {
    let batch_sz = cfg.batch_sz;
    let max_stride = cfg.max_stride;
    let mut h_buf = Vec::with_capacity(batch_sz * max_stride);
    h_buf.resize(batch_sz * max_stride, 0.0);
    let tmp_buf = DeviceMem::zeros(batch_sz * max_stride, stream.conn());
    let out = DeviceOutput::new(batch_sz, max_stride, cap, stream.conn());
    Rc::new(RefCell::new(DeviceVarInputOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      stream:   stream,
      out:      out,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      r_state:  vec![],
      in_dims:  Vec::with_capacity(batch_sz),
      tmp_dims: Vec::with_capacity(batch_sz),
      h_buf:    h_buf,
      tmp_buf:  tmp_buf,
      _marker:  PhantomData,
    }))
  }
}

impl<S> Operator for DeviceVarInputOperator<S> /*where S: SampleDatum<[f32]>*/ {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> DeviceOperator for DeviceVarInputOperator<S> /*where S: SampleDatum<[f32]>*/ {
  fn _output(&self, arm: usize) -> DeviceOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

/*impl<S> NewDiffOperator<S> for DeviceVarInputOperator<S> where S: SampleDatum<[f32]> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
  }

  fn _save_rng_state(&mut self) {
    self.r_state.clear();
    self.r_state.resize(self.rng.state_size(), 0);
    self.rng.extract_state(&mut self.r_state);
  }

  fn _restore_rng_state(&mut self) {
    self.rng.set_state(&self.r_state);
  }

  fn _load_batch(&mut self, samples: &[S]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    self.out.batch_sz.set(batch_size);

    self.h_buf.reshape_mut(batch_size * self.cfg.max_stride).set_constant(0.0);
    self.in_dims.clear();
    for (idx, sample) in samples.iter().enumerate() {
      // FIXME(20160920): check input shape.
      /*assert_eq!(self.cfg.max_stride, sample.input.dim().flat_len());
      assert_eq!(sample.input.stride(), sample.input.dim().least_stride());*/
      if let Err(_) = sample.extract_input(&mut self.h_buf[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride]) {
        panic!("DeviceVarInputOperator: failed to extract input");
      }
      if let Some(Shape3d(in_dim)) = sample.shape() {
        self.in_dims.push(in_dim);
      } else {
        panic!();
      }
    }
    // FIXME(20161014): could also do asynchronous loads in the previous loop.
    self.out.buf.borrow_mut().as_mut()
      .slice_mut(0, batch_size * self.cfg.max_stride)
      .load_sync(&self.h_buf[ .. batch_size * self.cfg.max_stride], self.stream.conn());
  }

  fn _forward(&mut self, phase: OpPhase) {
    let batch_size = self.out.batch_sz.get();
    self.tmp_dims.clear();
    for idx in 0 .. batch_size {
      self.tmp_dims.push(self.in_dims[idx]);
    }
    let mut out_buf = self.out.buf.borrow_mut();
    for preproc in self.cfg.preprocs.iter() {
      match preproc {
        &VarInputPreproc::Scale{scale} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            out_buf.as_mut()
              .slice_mut(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride)
              .reshape_mut(dim.flat_len())
              .scale(scale, self.stream.conn());
          }
        }
        &VarInputPreproc::ChannelShift{ref shift} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            let space_len = dim.0 * dim.1;
            for a in 0 .. self.cfg.out_dim.2 {
              out_buf.as_mut()
                .slice_mut(a * space_len + idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride)
                .reshape_mut(space_len)
                .add_scalar(-shift[a], self.stream.conn());
            }
          }
        }
        &VarInputPreproc::RandomResize2d{lo, hi, ref phases} => {
          if phases.contains(&phase) {
            //let mut out_buf = self.out.buf.borrow_mut();
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              let resized_out_d = self.rng.gen_range(lo, hi+1);
              let (out_w, out_h) = if in_dim.0 >= in_dim.1 {
                let sy = resized_out_d as f64 / in_dim.1 as f64;
                ((sy * in_dim.0 as f64).round() as usize, resized_out_d)
              } else {
                let sx = resized_out_d as f64 / in_dim.0 as f64;
                (resized_out_d, (sx * in_dim.1 as f64).round() as usize)
              };
              let out_dim = (out_w, out_h, in_dim.2);
              let out_len = out_dim.flat_len();
              {
                let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride);
                let tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
                unsafe { neuralops_cuda_interpolate2d_catmullrom(
                    out.as_ptr(),
                    in_dim.0, in_dim.1, in_dim.2,
                    tmp.as_mut_ptr(),
                    out_dim.0, out_dim.1,
                    self.stream.conn().raw_stream().ptr,
                ) };
              }
              let tmp = self.tmp_buf.as_ref().slice(idx * out_len, (idx+1) * out_len);
              let mut out = out_buf.as_mut().slice_mut(idx * self.cfg.max_stride, idx * self.cfg.max_stride + out_len);
              out.copy(tmp, self.stream.conn());
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::RandomCrop2d{crop_w, crop_h, pad_w, pad_h, ref phases} => {
          if phases.contains(&phase) {
            //let mut out_buf = self.out.buf.borrow_mut();
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              assert!(crop_w <= in_dim.0 + 2 * pad_w);
              assert!(crop_h <= in_dim.1 + 2 * pad_h);
              let out_dim = (crop_w, crop_h, in_dim.2);
              let out_len = out_dim.flat_len();
              let offset_x = self.rng.gen_range(0, in_dim.0 + 2 * pad_w - crop_w + 1) as isize - pad_w as isize;
              let offset_y = self.rng.gen_range(0, in_dim.1 + 2 * pad_h - crop_h + 1) as isize - pad_h as isize;
              {
                let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride);
                let tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
                unsafe { neuralops_cuda_image2d_crop(
                    out.as_ptr(),
                    in_dim.0, in_dim.1, in_dim.2,
                    offset_x, offset_y,
                    tmp.as_mut_ptr(),
                    out_dim.0, out_dim.1,
                    self.stream.conn().raw_stream().ptr,
                ) };
              }
              let tmp = self.tmp_buf.as_ref().slice(idx * out_len, (idx+1) * out_len);
              let mut out = out_buf.as_mut().slice_mut(idx * self.cfg.max_stride, idx * self.cfg.max_stride + out_len);
              out.copy(tmp, self.stream.conn());
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::CenterCrop2d{crop_w, crop_h, ref phases} => {
          if phases.contains(&phase) {
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              assert!(crop_w <= in_dim.0);
              assert!(crop_h <= in_dim.1);
              let out_dim = (crop_w, crop_h, in_dim.2);
              let out_len = out_dim.flat_len();
              let offset_x = ((in_dim.0 - crop_w) / 2) as isize;
              let offset_y = ((in_dim.1 - crop_h) / 2) as isize;
              {
                let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride);
                let tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
                unsafe { neuralops_cuda_image2d_crop(
                    out.as_ptr(),
                    in_dim.0, in_dim.1, in_dim.2,
                    offset_x, offset_y,
                    tmp.as_mut_ptr(),
                    out_dim.0, out_dim.1,
                    self.stream.conn().raw_stream().ptr,
                ) };
              }
              let tmp = self.tmp_buf.as_ref().slice(idx * out_len, (idx+1) * out_len);
              let mut out = out_buf.as_mut().slice_mut(idx * self.cfg.max_stride, idx * self.cfg.max_stride + out_len);
              out.copy(tmp, self.stream.conn());
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::RandomFlipX{ref phases} => {
          if phases.contains(&phase) {
            for idx in 0 .. batch_size {
              let out_dim = self.tmp_dims[idx];
              let out_len = out_dim.flat_len();
              let bernoulli = self.rng.gen_range(0, 2);
              match bernoulli {
                0 => {}
                1 => {
                  {
                    let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride);
                    let tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
                    unsafe { neuralops_cuda_image2d_flip(
                        out.as_ptr(),
                        out_dim.0, out_dim.1, out_dim.2,
                        tmp.as_mut_ptr(),
                        self.stream.conn().raw_stream().ptr,
                    ) };
                  }
                  let tmp = self.tmp_buf.as_ref().slice(idx * out_len, (idx+1) * out_len);
                  let mut out = out_buf.as_mut().slice_mut(idx * self.cfg.max_stride, idx * self.cfg.max_stride + out_len);
                  out.copy(tmp, self.stream.conn());
                }
                _ => unreachable!(),
              }
            }
          }
        }
        _ => unimplemented!(),
      }
    }
    let out_len = self.cfg.out_dim.flat_len();
    for idx in 0 .. batch_size {
      assert_eq!(self.cfg.out_dim, self.tmp_dims[idx]);
      let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride);
      let mut tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
      tmp.copy(out, self.stream.conn());
    }
    let tmp = self.tmp_buf.as_ref().slice(0, batch_size * out_len);
    let mut out = out_buf.as_mut().slice_mut(0, batch_size * out_len);
    out.copy(tmp, self.stream.conn());
  }

  fn _backward(&mut self) {
  }
}*/

impl NewDiffOperator<SampleItem> for DeviceVarInputOperator<SampleItem> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<SampleItem, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<SampleItem, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
  }

  fn _save_rng_state(&mut self) {
    self.r_state.clear();
    self.r_state.resize(self.rng.state_size(), 0);
    self.rng.extract_state(&mut self.r_state);
  }

  fn _restore_rng_state(&mut self) {
    self.rng.set_state(&self.r_state);
  }

  fn _load_batch(&mut self, samples: &[SampleItem]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    self.out.batch_sz.set(batch_size);

    self.h_buf.reshape_mut(batch_size * self.cfg.max_stride).set_constant(0.0);
    self.in_dims.clear();
    for (idx, sample) in samples.iter().enumerate() {
      if sample.kvs.contains::<SampleSharedExtractInputKey<[f32]>>() {
        let data = sample.kvs.get::<SampleSharedExtractInputKey<[f32]>>().unwrap();
        data.extract_input(&mut self.h_buf[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride]).unwrap();
      } else if sample.kvs.contains::<SampleExtractInputKey<[f32]>>() {
        let data = sample.kvs.get::<SampleExtractInputKey<[f32]>>().unwrap();
        data.extract_input(&mut self.h_buf[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride]).unwrap();
      } else {
        panic!();
      }
      if sample.kvs.contains::<SampleInputShape3dKey>() {
        let in_dim = *sample.kvs.get::<SampleInputShape3dKey>().unwrap();
        self.in_dims.push(in_dim);
      } else {
        panic!();
      }
      /*if let Err(_) = sample.extract_input(&mut self.h_buf[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride]) {
        panic!("DeviceVarInputOperator: failed to extract input");
      }
      if let Some(Shape3d(in_dim)) = sample.shape() {
        self.in_dims.push(in_dim);
      } else {
        panic!();
      }*/
    }
    // FIXME(20161014): could also do asynchronous loads in the previous loop.
    self.out.buf.borrow_mut().as_mut()
      .slice_mut(0, batch_size * self.cfg.max_stride)
      .load_sync(&self.h_buf[ .. batch_size * self.cfg.max_stride], self.stream.conn());
  }

  fn _forward(&mut self, phase: OpPhase) {
    let batch_size = self.out.batch_sz.get();
    self.tmp_dims.clear();
    for idx in 0 .. batch_size {
      self.tmp_dims.push(self.in_dims[idx]);
    }
    let mut out_buf = self.out.buf.borrow_mut();
    for preproc in self.cfg.preprocs.iter() {
      match preproc {
        &VarInputPreproc::Scale{scale} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            out_buf.as_mut()
              .slice_mut(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride)
              .reshape_mut(dim.flat_len())
              .scale(scale, self.stream.conn());
          }
        }
        &VarInputPreproc::DivScale{scale} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            out_buf.as_mut()
              .slice_mut(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride)
              .reshape_mut(dim.flat_len())
              .div_scalar(scale, self.stream.conn());
          }
        }
        &VarInputPreproc::ChannelShift{ref shift} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            let space_len = dim.0 * dim.1;
            for a in 0 .. self.cfg.out_dim.2 {
              out_buf.as_mut()
                .slice_mut(a * space_len + idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride)
                .reshape_mut(space_len)
                .add_scalar(-shift[a], self.stream.conn());
            }
          }
        }
        &VarInputPreproc::RandomResize2d{lo, hi, ref phases} => {
          if phases.contains(&phase) {
            //let mut out_buf = self.out.buf.borrow_mut();
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              let resized_out_d = self.rng.gen_range(lo, hi+1);
              let (out_w, out_h) = if in_dim.0 >= in_dim.1 {
                let sy = resized_out_d as f64 / in_dim.1 as f64;
                ((sy * in_dim.0 as f64).round() as usize, resized_out_d)
              } else {
                let sx = resized_out_d as f64 / in_dim.0 as f64;
                (resized_out_d, (sx * in_dim.1 as f64).round() as usize)
              };
              let out_dim = (out_w, out_h, in_dim.2);
              let out_len = out_dim.flat_len();
              {
                let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride);
                let tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
                unsafe { neuralops_cuda_interpolate2d_catmullrom(
                    out.as_ptr(),
                    in_dim.0, in_dim.1, in_dim.2,
                    tmp.as_mut_ptr(),
                    out_dim.0, out_dim.1,
                    self.stream.conn().raw_stream().ptr,
                ) };
              }
              let tmp = self.tmp_buf.as_ref().slice(idx * out_len, (idx+1) * out_len);
              let mut out = out_buf.as_mut().slice_mut(idx * self.cfg.max_stride, idx * self.cfg.max_stride + out_len);
              out.copy(tmp, self.stream.conn());
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::RandomCrop2d{crop_w, crop_h, pad_w, pad_h, ref phases} => {
          if phases.contains(&phase) {
            //let mut out_buf = self.out.buf.borrow_mut();
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              assert!(crop_w <= in_dim.0 + 2 * pad_w);
              assert!(crop_h <= in_dim.1 + 2 * pad_h);
              let out_dim = (crop_w, crop_h, in_dim.2);
              let out_len = out_dim.flat_len();
              let offset_x = self.rng.gen_range(0, in_dim.0 + 2 * pad_w - crop_w + 1) as isize - pad_w as isize;
              let offset_y = self.rng.gen_range(0, in_dim.1 + 2 * pad_h - crop_h + 1) as isize - pad_h as isize;
              {
                let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride);
                let tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
                unsafe { neuralops_cuda_image2d_crop(
                    out.as_ptr(),
                    in_dim.0, in_dim.1, in_dim.2,
                    offset_x, offset_y,
                    tmp.as_mut_ptr(),
                    out_dim.0, out_dim.1,
                    self.stream.conn().raw_stream().ptr,
                ) };
              }
              let tmp = self.tmp_buf.as_ref().slice(idx * out_len, (idx+1) * out_len);
              let mut out = out_buf.as_mut().slice_mut(idx * self.cfg.max_stride, idx * self.cfg.max_stride + out_len);
              out.copy(tmp, self.stream.conn());
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::OffsetCrop2d{crop_w, crop_h, offset_x, offset_y, ref phases} => {
          if phases.contains(&phase) {
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              let out_dim = (crop_w, crop_h, in_dim.2);
              let out_len = out_dim.flat_len();
              {
                let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride);
                let tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
                unsafe { neuralops_cuda_image2d_crop(
                    out.as_ptr(),
                    in_dim.0, in_dim.1, in_dim.2,
                    offset_x, offset_y,
                    tmp.as_mut_ptr(),
                    out_dim.0, out_dim.1,
                    self.stream.conn().raw_stream().ptr,
                ) };
              }
              let tmp = self.tmp_buf.as_ref().slice(idx * out_len, (idx+1) * out_len);
              let mut out = out_buf.as_mut().slice_mut(idx * self.cfg.max_stride, idx * self.cfg.max_stride + out_len);
              out.copy(tmp, self.stream.conn());
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::CenterCrop2d{crop_w, crop_h, ref phases} => {
          if phases.contains(&phase) {
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              assert!(crop_w <= in_dim.0);
              assert!(crop_h <= in_dim.1);
              let out_dim = (crop_w, crop_h, in_dim.2);
              let out_len = out_dim.flat_len();
              let offset_x = ((in_dim.0 - crop_w) / 2) as isize;
              let offset_y = ((in_dim.1 - crop_h) / 2) as isize;
              {
                let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride);
                let tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
                unsafe { neuralops_cuda_image2d_crop(
                    out.as_ptr(),
                    in_dim.0, in_dim.1, in_dim.2,
                    offset_x, offset_y,
                    tmp.as_mut_ptr(),
                    out_dim.0, out_dim.1,
                    self.stream.conn().raw_stream().ptr,
                ) };
              }
              let tmp = self.tmp_buf.as_ref().slice(idx * out_len, (idx+1) * out_len);
              let mut out = out_buf.as_mut().slice_mut(idx * self.cfg.max_stride, idx * self.cfg.max_stride + out_len);
              out.copy(tmp, self.stream.conn());
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::RandomFlipX{ref phases} => {
          if phases.contains(&phase) {
            for idx in 0 .. batch_size {
              let out_dim = self.tmp_dims[idx];
              let out_len = out_dim.flat_len();
              let bernoulli = self.rng.gen_range(0, 2);
              match bernoulli {
                0 => {}
                1 => {
                  {
                    let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, (idx+1) * self.cfg.max_stride);
                    let tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
                    unsafe { neuralops_cuda_image2d_flip(
                        out.as_ptr(),
                        out_dim.0, out_dim.1, out_dim.2,
                        tmp.as_mut_ptr(),
                        self.stream.conn().raw_stream().ptr,
                    ) };
                  }
                  let tmp = self.tmp_buf.as_ref().slice(idx * out_len, (idx+1) * out_len);
                  let mut out = out_buf.as_mut().slice_mut(idx * self.cfg.max_stride, idx * self.cfg.max_stride + out_len);
                  out.copy(tmp, self.stream.conn());
                }
                _ => unreachable!(),
              }
            }
          }
        }
        _ => unimplemented!(),
      }
    }
    let out_len = self.cfg.out_dim.flat_len();
    assert!(out_len <= self.cfg.max_stride);
    for idx in 0 .. batch_size {
      assert_eq!(self.cfg.out_dim, self.tmp_dims[idx]);
      let out = out_buf.as_ref().slice(idx * self.cfg.max_stride, idx * self.cfg.max_stride + out_len);
      let mut tmp = self.tmp_buf.as_mut().slice_mut(idx * out_len, (idx+1) * out_len);
      tmp.copy(out, self.stream.conn());
    }
    let tmp = self.tmp_buf.as_ref().slice(0, batch_size * out_len);
    let mut out = out_buf.as_mut().slice_mut(0, batch_size * out_len);
    out.copy(tmp, self.stream.conn());
  }

  fn _backward(&mut self) {
  }
}
