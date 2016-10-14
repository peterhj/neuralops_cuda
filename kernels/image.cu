#include <cuda_runtime_api.h>
#include <stddef.h>

__global__ void image2d_crop(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    int x_offset,
    int y_offset,
    float *out_pixels,
    int crop_width,
    int crop_height)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int u = idx % crop_width;
  int v = (idx / crop_width) % crop_height;
  int c = idx / (crop_width * crop_height);
  int n = crop_width * crop_height * channels;
  if (idx < n) {
    int x = u + x_offset;
    int y = v + y_offset;
    if ((x >= 0) && (x < crop_width) && (y >= 0) && (y < crop_height) && (c < channels)) {
      int in_idx = x + y * in_width + c * in_width * in_height;
      out_pixels[idx] = in_pixels[in_idx];
    } else {
      out_pixels[idx] = 0.0f;
    }
  }
}

extern "C" void neuralops_cuda_image2d_crop(
    const float *in_pixels,
    size_t in_width,
    size_t in_height,
    size_t channels,
    ptrdiff_t x_offset,
    ptrdiff_t y_offset,
    float *out_pixels,
    size_t crop_width,
    size_t crop_height,
    cudaStream_t stream)
{
  int n = crop_width * crop_height * channels;
  image2d_crop<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_pixels,
      in_width,
      in_height,
      channels,
      x_offset,
      y_offset,
      out_pixels,
      crop_width,
      crop_height);
}

__global__ void image2d_flip(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    float *out_pixels)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int x = idx % in_width;
  int y = (idx / in_width) % in_height;
  int c = idx / (in_width * in_height);
  int n = in_width * in_height * channels;
  if (idx < n) {
    int in_idx = (in_width - x - 1) + y * in_width + c * in_width * in_height;
    out_pixels[idx] = in_pixels[in_idx];
  }
}

extern "C" void neuralops_cuda_image2d_flip(
    const float *in_pixels,
    size_t in_width,
    size_t in_height,
    size_t channels,
    float *out_pixels,
    cudaStream_t stream)
{
  int n = in_width * in_height * channels;
  image2d_flip<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_pixels,
      in_width,
      in_height,
      channels,
      out_pixels);
}
