use devicemem_cuda::prelude::*;
use neuralops::prelude::*;
use operator::prelude::*;

//use typemap::{ShareMap, TypeMap, Key};

use std::u32;
use std::cell::{RefCell};
use std::collections::{HashMap};
use std::hash::{Hash};
use std::rc::{Rc};

pub struct DeviceCachedSampleItem<K> {
  cache:    Rc<DeviceSampleCache<K>>,
  idx:      usize,
  label:    bool,
  target:   bool,
  weight:   bool,
}

impl<K> DeviceCachedSampleItem<K> {
  pub fn has_class_label(&self) -> bool {
    self.label
  }

  pub fn has_regress_target(&self) -> bool {
    self.target
  }

  pub fn has_weight(&self) -> bool {
    self.weight
  }
}

impl<K> SampleInputShape<(usize, usize, usize)> for DeviceCachedSampleItem<K> {
  fn input_shape(&self) -> Option<(usize, usize, usize)> {
    let inner = self.cache.inner.borrow();
    Some(inner.input_shape[self.idx])
  }
}

pub trait DeviceSampleExtractInput<U: ?Sized> {
  fn device_extract_input(&self, output: &mut DeviceMem<f32>);
}

pub trait DeviceSampleExtractLabel<U: ?Sized> {
  fn device_extract_label(&self, output: &mut DeviceMem<u32>);
}

pub trait DeviceSampleExtractTarget<U: ?Sized> {
  fn device_extract_target(&self, output: &mut DeviceMem<f32>);
}

pub trait DeviceSampleExtractWeight<U: ?Sized> {
  fn device_extract_weight(&self, output: &mut DeviceMem<f32>);
}

#[derive(Clone)]
pub struct SampleCacheConfig {
  pub batch_sz:     usize,
  pub capacity:     usize,
  pub input_stride: usize,
  pub input_dtype:  Dtype,
}

pub struct DeviceSampleCache<K> {
  cfg:      SampleCacheConfig,
  inner:    RefCell<DeviceSampleCacheInner<K>>,
}

pub struct DeviceSampleCacheInner<K> {
  count:        usize,
  batch_count:  usize,
  flush_count:  usize,
  key_map:      HashMap<K, usize>,
  input_shape:  Vec<(usize, usize, usize)>,
  input_buf:    DeviceMem<u8>,
  label_buf:    DeviceMem<u32>,
  target_buf:   DeviceMem<f32>,
  weight_buf:   DeviceMem<f32>,
  input_buf_h:  AsyncMem<u8>,
  label_buf_h:  AsyncMem<u32>,
  target_buf_h: AsyncMem<f32>,
  weight_buf_h: AsyncMem<f32>,
}

impl<K> DeviceSampleCache<K> where K: Clone + Eq + Hash {
  pub fn new() -> Rc<DeviceSampleCache<K>> {
    unimplemented!();
  }

  pub fn clear(&self) {
    let mut inner = self.inner.borrow_mut();
    inner.count = 0;
    inner.batch_count = 0;
    inner.flush_count = 0;
    inner.key_map.clear();
  }

  pub fn flush(&self) {
    let mut inner = self.inner.borrow_mut();
    let n = inner.batch_count;
    inner.batch_count = 0;
    inner.flush_count += n;
    assert_eq!(inner.flush_count, inner.count);
  }

  pub fn append(&self, samples: &[(K, SampleItem)]) {
    let mut inner = self.inner.borrow_mut();
    for &(ref key, ref sample) in samples.iter() {
      let idx = inner.count;
      assert!(idx < self.cfg.capacity);
      inner.key_map.insert(key.clone(), idx);
      match self.cfg.input_dtype {
        Dtype::F32 => {
          if let Some(data) = sample.kvs.get::<SampleSharedExtractInputKey<[f32]>>() {
            //data.extract_input(h_buf).unwrap();
          } else if let Some(data) = sample.kvs.get::<SampleExtractInputKey<[f32]>>() {
            //data.extract_input(h_buf).unwrap();
          } else {
            panic!("SampleItem is missing [f32] data");
          }
        }
        Dtype::U8 => {
          if let Some(data) = sample.kvs.get::<SampleSharedExtractInputKey<[u8]>>() {
            //data.extract_input(h_buf).unwrap();
          } else if let Some(data) = sample.kvs.get::<SampleExtractInputKey<[u8]>>() {
            //data.extract_input(h_buf).unwrap();
          } else {
            panic!("SampleItem is missing [u8] data");
          }
        }
        _ => unimplemented!(),
      }
      if sample.kvs.contains::<SampleClassLabelKey>() {
        let label = *sample.kvs.get::<SampleClassLabelKey>().unwrap();
        inner.label_buf_h.as_mut()[idx] = label;
      } else {
        inner.label_buf_h.as_mut()[idx] = u32::MAX;
      }
      if sample.kvs.contains::<SampleRegressTargetKey>() {
        let target = *sample.kvs.get::<SampleRegressTargetKey>().unwrap();
        inner.target_buf_h.as_mut()[idx] = target;
      } else {
        inner.target_buf_h.as_mut()[idx] = 0.0;
      }
      if sample.kvs.contains::<SampleWeightKey>() {
        let weight = *sample.kvs.get::<SampleWeightKey>().unwrap();
        inner.weight_buf_h.as_mut()[idx] = weight;
      } else {
        inner.weight_buf_h.as_mut()[idx] = 1.0;
      }
      // TODO
      inner.count += 1;
      inner.batch_count += 1;
      if inner.batch_count == self.cfg.batch_sz {
        self.flush();
      }
    }
  }

  pub fn query(&self, key: K) -> DeviceCachedSampleItem<K> {
    unimplemented!();
  }
}
