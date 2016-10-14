extern crate gcc;

fn main() {
  gcc::Config::new()
    .compiler("/usr/local/cuda/bin/nvcc")
    .opt_level(3)
    // FIXME(20151207): for working w/ K80.
    //.flag("-arch=sm_37")
    .flag("-arch=sm_52")
    .flag("-Xcompiler")
    .flag("\'-fPIC\'")
    .include("/usr/local/cuda/include")
    .file("kernels/image.cu")
    .file("kernels/interpolate.cu")
    .compile("libneuralops_cuda_kernels.a");

  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64");
}
