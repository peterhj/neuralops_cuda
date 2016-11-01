use std::fs::{File};
use std::io::{Write};
use std::mem::{size_of};
use std::path::{Path};
use std::slice::{from_raw_parts};

pub fn dump_to_file<T>(path: &Path, data: &[T]) where T: Copy {
  let mut file = File::create(path).unwrap();
  let raw_data: &[u8] = unsafe { from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<T>()) };
  file.write_all(raw_data).unwrap();
}
