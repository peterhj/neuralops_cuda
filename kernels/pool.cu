#include <cuda_runtime_api.h>

__global__ void AvePoolForward(const int nthreads,
    const float* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* const top_data) {
  //CUDA_KERNEL_LOOP(index, nthreads) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    float aveval = 0;
    const float* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

extern "C" void neuralops_cuda_caffe_avgpool2d_fwd(
    const float* bottom_data,
    int num, int channels_, int height_, int width_,
    int pooled_height_, int pooled_width_,
    int kernel_h_, int kernel_w_,
    int pad_h_, int pad_w_,
    int stride_h_, int stride_w_,
    float* top_data,
    cudaStream_t stream)
{
  int count = pooled_width_ * pooled_height_ * channels_ * num;
  AvePoolForward<<<(count+1024-1)/1024, 1024, 0, stream>>>(
      count, bottom_data, num, channels_,
      height_, width_, pooled_height_, pooled_width_, kernel_h_,
      kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
}

__global__ void AvePoolBackward(const int nthreads, const float* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    float* const bottom_diff) {
  //CUDA_KERNEL_LOOP(index, nthreads) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    float gradient = 0;
    const float* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

extern "C" void neuralops_cuda_caffe_avgpool2d_bwd(
    const float* top_diff,
    int num, int channels_, int height_, int width_,
    int pooled_height_, int pooled_width_,
    int kernel_h_, int kernel_w_,
    int pad_h_, int pad_w_,
    int stride_h_, int stride_w_,
    float *bottom_diff,
    cudaStream_t stream)
{
  int count = width_ * height_ * channels_ * num;
  AvePoolBackward<<<(count+1024-1)/1024, 1024, 0, stream>>>(
      count, top_diff, num, channels_,
      height_, width_, pooled_height_, pooled_width_, kernel_h_,
      kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
}
