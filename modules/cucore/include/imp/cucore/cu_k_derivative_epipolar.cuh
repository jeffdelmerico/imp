#ifndef IMP_CU_K_DERIVATIVE_CUH
#define IMP_CU_K_DERIVATIVE_CUH

#include <cuda_runtime.h>
#include <imp/cucore/cu_texture.cuh>

namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
/** compute forward differences in x- and y- direction */
static __device__ __forceinline__ float2 dpEpi(
    const imp::cu::Texture2D& tex, float2 epi_vec, size_t x, size_t y)
{

  float2 xc = make_float2((float)x, (float)y);
  float2 epi_vec_n = make_float2(-epi_vec.y, epi_vec.x);
  float2 x_xp = xc + epi_vec;
  float2 x_yp = xc + epi_vec_n;

  float2 grad = make_float2(0.0f, 0.0f);
  float cval = tex.fetch<float>(xc.x, xc.y);
  grad.x = tex.fetch<float>(x_xp.x, x_xp.y) - cval;
  grad.x = tex.fetch<float>(x_yp.x, x_yp.y) - cval;
  return grad;
}

//-----------------------------------------------------------------------------
/** compute divergence using backward differences (adjugate from dp). */
static __device__ __forceinline__
float dpAdEpi(const imp::cu::Texture2D& tex, float2 epi_vec,
           size_t x, size_t y, size_t width, size_t height)
{

  float2 xc = make_float2((float)x, (float)y);
  float2 epi_vec_n = make_float2(-epi_vec.y, epi_vec.x);
  float2 x_xm = xc - epi_vec;
  float2 x_ym = xc - epi_vec_n;


  float2 c = tex.fetch<float2>(xc.x,xc.y);
  float2 w = tex.fetch<float2>(x_xm.x, x_xm.y);
  float2 n = tex.fetch<float2>(x_ym.x, x_ym.y);

  if (x == 0)
    w.x = 0.0f;
  else if (x >= width-1)
    c.x = 0.0f;

  if (y == 0)
    n.y = 0.0f;
  else if (y >= height-1)
    c.y = 0.0f;

  return (c.x - w.x + c.y - n.y);
}

//-----------------------------------------------------------------------------
/** compute weighted divergence using backward differences (adjugate from dp). */
static __device__ __forceinline__
float dpAdWeighted(const imp::cu::Texture2D& tex, const imp::cu::Texture2D& g_tex,
                   size_t x, size_t y, size_t width, size_t height)
{
  float2 c = tex.fetch<float2>(x,y);
  float2 w = tex.fetch<float2>(x-1, y);
  float2 n = tex.fetch<float2>(x, y-1);

  float g = g_tex.fetch<float>(x,y);
  float g_w = g_tex.fetch<float>(x-1, y);
  float g_n = g_tex.fetch<float>(x, y-1);

  if (x == 0)
    w.x = 0.0f;
  else if (x >= width-1)
    c.x = 0.0f;

  if (y == 0)
    n.y = 0.0f;
  else if (y >= height-1)
    c.y = 0.0f;

  return (c.x*g - w.x*g_w + c.y*g - n.y*g_n);
}

} // namespace cu
} // namespace imp

#endif // IMP_CU_K_DERIVATIVE_CUH

