#include <cuda_runtime_api.h>
#include <stdint.h>

__device__ float tex2d_clamp_uvc(const float *pixels, int width, int height, int u, int v, int c) {
  int clamp_u = min(max(0, u), width-1);
  int clamp_v = min(max(0, v), height-1);
  //return pixels[clamp_u + clamp_v * width + c * width * height];
  return pixels[clamp_u + width * (clamp_v + height * c)];
}

__device__ float tex2d_clamp_cuv(const float *pixels, int channels, int width, int height, int c, int u, int v) {
  int clamp_u = min(max(0, u), width-1);
  int clamp_v = min(max(0, v), height-1);
  return pixels[c + channels * (clamp_u + width * clamp_v)];
}

__device__ float lerp_filter(float a, float b, float t)
{
  return a + t * (a - b);
}

__device__ float bicubic_w0(float a) {
  return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);
}

__device__ float bicubic_w1(float a) {
  return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__device__ float bicubic_w2(float a) {
  return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__device__ float bicubic_w3(float a) {
  return (1.0f/6.0f)*(a*a*a);
}

__device__ float catrom_w0(float a) {
    //return -0.5f*a + a*a - 0.5f*a*a*a;
    return a*(-0.5f + a*(1.0f - 0.5f*a));
}

__device__ float catrom_w1(float a) {
    //return 1.0f - 2.5f*a*a + 1.5f*a*a*a;
    return 1.0f + a*a*(-2.5f + 1.5f*a);
}

__device__ float catrom_w2(float a) {
    //return 0.5f*a + 2.0f*a*a - 1.5f*a*a*a;
    return a*(0.5f + a*(2.0f - 1.5f*a));
}

__device__ float catrom_w3(float a) {
    //return -0.5f*a*a + 0.5f*a*a*a;
    return a*a*(-0.5f + 0.5f*a);
}

/*__device__ float mitchell_w0(float a) {
  float b = absf(a);
  return
      (b < 1.0f) * () +
      (b >= 1.0f) * (b < 2.0f) * ();
}*/

__device__ float interpolate_bicubic_filter(
    float x,
    float a0,
    float a1,
    float a2,
    float a3)
{
  float r = a0 * bicubic_w0(x);
  r += a1 * bicubic_w1(x);
  r += a2 * bicubic_w2(x);
  r += a3 * bicubic_w3(x);
  return r;
}

__device__ float interpolate_bicubic_interpolate(
    const float *pixels,
    int width,
    int height,
    float u,
    float v,
    int c)
{
  u -= 0.5f;
  v -= 0.5f;
  float px = floorf(u);
  float py = floorf(v);
  float fx = u - px;
  float fy = v - py;
  int ipx = (int)px;
  int ipy = (int)py;
  return interpolate_bicubic_filter(fy,
      interpolate_bicubic_filter(fx,
          tex2d_clamp_uvc(pixels, width, height, ipx-1, ipy-1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx,   ipy-1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy-1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+2, ipy-1, c)),
      interpolate_bicubic_filter(fx,
          tex2d_clamp_uvc(pixels, width, height, ipx-1, ipy,   c),
          tex2d_clamp_uvc(pixels, width, height, ipx,   ipy,   c),
          tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy,   c),
          tex2d_clamp_uvc(pixels, width, height, ipx+2, ipy,   c)),
      interpolate_bicubic_filter(fx,
          tex2d_clamp_uvc(pixels, width, height, ipx-1, ipy+1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx,   ipy+1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy+1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+2, ipy+1, c)),
      interpolate_bicubic_filter(fx,
          tex2d_clamp_uvc(pixels, width, height, ipx-1, ipy+2, c),
          tex2d_clamp_uvc(pixels, width, height, ipx,   ipy+2, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy+2, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+2, ipy+2, c)));
}

__global__ void interpolate_bicubic_kernel(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    float *out_pixels,
    int out_width,
    int out_height)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int x = idx % out_width;
  int y = (idx / out_width) % out_height;
  int c = idx / (out_width * out_height);

  if ((x < out_width) && (y < out_height) && (c < channels)) {
    float u = ((float)x) / ((float)out_width) * ((float)in_width);
    float v = ((float)y) / ((float)out_height) * ((float)in_height);

    float interp_value = interpolate_bicubic_interpolate(in_pixels, in_width, in_height, u, v, c);

    out_pixels[x + y * out_width + c * out_width * out_height] = interp_value;
  }
}

extern "C" void neuralops_cuda_interpolate2d_bicubic(
    const float *in_pixels,
    size_t in_width,
    size_t in_height,
    size_t channels,
    float *out_pixels,
    size_t out_width,
    size_t out_height,
    cudaStream_t stream)
{
  int n = out_width * out_height * channels;
  interpolate_bicubic_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_pixels,
      in_width,
      in_height,
      channels,
      out_pixels,
      out_width,
      out_height);
}

__device__ float catmullrom_filter(
    float x,
    float a0,
    float a1,
    float a2,
    float a3)
{
  float r = a0 * catrom_w0(x);
  r += a1 * catrom_w1(x);
  r += a2 * catrom_w2(x);
  r += a3 * catrom_w3(x);
  return r;
}

__device__ float catmullrom_filter2d(
    const float *pixels,
    int width,
    int height,
    float u,
    float v,
    int c)
{
  u -= 0.5f;
  v -= 0.5f;
  float px = floorf(u);
  float py = floorf(v);
  float fx = u - px;
  float fy = v - py;
  int ipx = (int)px;
  int ipy = (int)py;
  return catmullrom_filter(fy,
      catmullrom_filter(fx,
          tex2d_clamp_uvc(pixels, width, height, ipx-1, ipy-1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx,   ipy-1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy-1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+2, ipy-1, c)),
      catmullrom_filter(fx,
          tex2d_clamp_uvc(pixels, width, height, ipx-1, ipy,   c),
          tex2d_clamp_uvc(pixels, width, height, ipx,   ipy,   c),
          tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy,   c),
          tex2d_clamp_uvc(pixels, width, height, ipx+2, ipy,   c)),
      catmullrom_filter(fx,
          tex2d_clamp_uvc(pixels, width, height, ipx-1, ipy+1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx,   ipy+1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy+1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+2, ipy+1, c)),
      catmullrom_filter(fx,
          tex2d_clamp_uvc(pixels, width, height, ipx-1, ipy+2, c),
          tex2d_clamp_uvc(pixels, width, height, ipx,   ipy+2, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy+2, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+2, ipy+2, c)));
}

__global__ void catmullrom_kernel(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    float *out_pixels,
    int out_width,
    int out_height)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int x = idx % out_width;
  int y = (idx / out_width) % out_height;
  int c = idx / (out_width * out_height);

  if ((x < out_width) && (y < out_height) && (c < channels)) {
    float u = ((float)x) / ((float)out_width) * ((float)in_width);
    float v = ((float)y) / ((float)out_height) * ((float)in_height);

    float interp_value = catmullrom_filter2d(in_pixels, in_width, in_height, u, v, c);

    out_pixels[x + y * out_width + c * out_width * out_height] = interp_value;
  }
}

extern "C" void neuralops_cuda_interpolate2d_catmullrom(
    const float *in_pixels,
    size_t in_width,
    size_t in_height,
    size_t channels,
    float *out_pixels,
    size_t out_width,
    size_t out_height,
    cudaStream_t stream)
{
  int n = out_width * out_height * channels;
  catmullrom_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_pixels,
      in_width,
      in_height,
      channels,
      out_pixels,
      out_width,
      out_height);
}

__device__ float interpolate_2x2_bilinear_interpolate(
    const float *pixels,
    int width,
    int height,
    float u,
    float v,
    int c)
{
  u -= 0.5f;
  v -= 0.5f;
  float px = floorf(u);
  float py = floorf(v);
  int ipx = (int)px;
  int ipy = (int)py;
  return 0.25 * (
      tex2d_clamp_uvc(pixels, width, height, ipx,   ipy,   c) +
      tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy,   c) +
      tex2d_clamp_uvc(pixels, width, height, ipx,   ipy+1, c) +
      tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy+1, c));
}

__global__ void interpolate_2x2_bilinear_kernel(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    float *out_pixels,
    int out_width,
    int out_height)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int x = idx % out_width;
  int y = (idx / out_width) % out_height;
  int c = idx / (out_width * out_height);

  if ((x < out_width) && (y < out_height) && (c < channels)) {
    float u = ((float)x) / ((float)out_width) * ((float)in_width);
    float v = ((float)y) / ((float)out_height) * ((float)in_height);

    float interp_value = interpolate_2x2_bilinear_interpolate(in_pixels, in_width, in_height, u, v, c);

    out_pixels[x + y * out_width + c * out_width * out_height] = interp_value;
  }
}

extern "C" void neuralops_cuda_interpolate2d_2x2_bilinear(
    const float *in_pixels,
    size_t in_width,
    size_t in_height,
    size_t channels,
    float *out_pixels,
    size_t out_width,
    size_t out_height,
    cudaStream_t stream)
{
  int n = out_width * out_height * channels;
  interpolate_2x2_bilinear_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_pixels,
      in_width,
      in_height,
      channels,
      out_pixels,
      out_width,
      out_height);
}

__device__ float interpolate_bilinear_interpolate(
    const float *pixels,
    int width,
    int height,
    float u,
    float v,
    int c)
{
  u -= 0.5f;
  v -= 0.5f;
  float px = floorf(u);
  float py = floorf(v);
  float fx = u - px;
  float fy = v - py;
  int ipx = (int)px;
  int ipy = (int)py;
  return lerp_filter(
      lerp_filter(
          tex2d_clamp_uvc(pixels, width, height, ipx,   ipy,   c),
          tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy,   c),
          fx),
      lerp_filter(
          tex2d_clamp_uvc(pixels, width, height, ipx,   ipy+1, c),
          tex2d_clamp_uvc(pixels, width, height, ipx+1, ipy+1, c),
          fx),
      fy);
}

__global__ void interpolate_bilinear_kernel(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    float *out_pixels,
    int out_width,
    int out_height)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int x = idx % out_width;
  int y = (idx / out_width) % out_height;
  int c = idx / (out_width * out_height);

  if ((x < out_width) && (y < out_height) && (c < channels)) {
    float u = ((float)x) / ((float)out_width) * ((float)in_width);
    float v = ((float)y) / ((float)out_height) * ((float)in_height);

    float interp_value = interpolate_bilinear_interpolate(in_pixels, in_width, in_height, u, v, c);

    out_pixels[x + y * out_width + c * out_width * out_height] = interp_value;
  }
}

extern "C" void neuralops_cuda_interpolate2d_bilinear(
    const float *in_pixels,
    size_t in_width,
    size_t in_height,
    size_t channels,
    float *out_pixels,
    size_t out_width,
    size_t out_height,
    cudaStream_t stream)
{
  int n = out_width * out_height * channels;
  interpolate_bilinear_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_pixels,
      in_width,
      in_height,
      channels,
      out_pixels,
      out_width,
      out_height);
}
