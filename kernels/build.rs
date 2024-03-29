extern crate gcc;

fn main() {
  gcc::Config::new()
    .compiler("/usr/local/cuda/bin/nvcc")
    .opt_level(3)
    // FIXME(20151207): for working w/ K80.
    //.flag("-arch=sm_37")
    .flag("-arch=sm_52")
    .flag("-prec-div=true")
    .flag("-prec-sqrt=true")
    .flag("-Xcompiler")
    .flag("\'-fno-strict-aliasing\'")
    .pic(true)
    .include("/usr/local/cuda/include")
    .file("activate.cu")
    .file("cast.cu")
    .file("clamp.cu")
    .file("conv.cu")
    .file("conv_batchnorm.cu")
    .file("gaussian.cu")
    .file("image.cu")
    .file("interpolate.cu")
    .file("linear.cu")
    .file("logistic.cu")
    .file("lstsq.cu")
    .file("map.cu")
    .file("pool.cu")
    .file("reduce.cu")
    .file("softmax.cu")
    .compile("libneuralops_cuda_kernels.a");

  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64");
}
