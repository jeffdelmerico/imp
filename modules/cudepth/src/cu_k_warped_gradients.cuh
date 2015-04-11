#ifndef IMP_CU_K_WARPED_GRADIENTS_CUH
#define IMP_CU_K_WARPED_GRADIENTS_CUH

#include <cuda_runtime_api.h>
#include <imp/core/types.hpp>
#include <imp/cuda_toolkit/helper_math.h>
#include <imp/cucore/cu_pinhole_camera.cuh>
#include <imp/cucore/cu_matrix.cuh>
#include <imp/cucore/cu_se3.cuh>


namespace imp {
namespace cu {

//------------------------------------------------------------------------------
template<typename Pixel>
__global__ void k_warpedGradients(Pixel* ix, Pixel* it, size_type stride,
                                  std::uint32_t width, std::uint32_t height,
                                  //                                  std::uint32_t roi_x, std::uint32_t roi_y,
                                  Texture2D i1_tex, Texture2D i2_tex, Texture2D u0_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x /*+ roi_x*/;
  const int y = blockIdx.y*blockDim.y + threadIdx.y /*+ roi_y*/;
  const int c = y*stride+x;

  if (x<width && y<height)
  {
    float u0 = u0_tex.fetch<float>(x,y);
    float wx = x+u0;

    float bd = .5f;
    if ((wx < bd) || (x < bd) || (wx > width-bd-1) || (x > width-bd-1) ||
        (y<bd) || (y>height-bd-1))
    {
      ix[c] =  0.0f;
      it[c] =  0.0f;
    }
    else
    {
      Pixel i1_c, i2_w_c, i2_w_m, i2_w_p;

      i1_tex.fetch(i1_c, x, y);

      i2_tex.fetch(i2_w_c, wx, y);
      i2_tex.fetch(i2_w_m, wx-0.5f, y);
      i2_tex.fetch(i2_w_p, wx+0.5f, y);

      // spatial gradient on warped image
      ix[c] = i2_w_p - i2_w_m;
      // temporal gradient between the warped moving image and the fixed image
      it[c] = i2_w_c - i1_c;
    }

  }
}

//------------------------------------------------------------------------------
template<typename Pixel>
__global__ void k_warpedGradientsEpipolarConstraint(
    Pixel* ix, Pixel* it, size_type stride,
//    Pixel32sC1* mask, size_type stride_32sC1,
    Pixel32fC2* d_p_mu, Pixel32fC2* d_epi_vec, size_type stride_32fC2,
    std::uint32_t width, std::uint32_t height,
    cu::PinholeCamera cam1, cu::PinholeCamera cam2,
    const cu::Matrix3f F_mov_fix, const cu::SE3<float> T_mov_fix,
    Texture2D i1_tex, Texture2D i2_tex, Texture2D u0_tex,
    Texture2D depth_proposal_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x /*+ roi_x*/;
  const int y = blockIdx.y*blockDim.y + threadIdx.y /*+ roi_y*/;
  const int c = y*stride+x;

  if (x<width && y<height)
  {
    // compute epipolar geometry
    float mu = depth_proposal_tex.fetch<float>(x,y);

    //    float sigma = sqrtf(depth_proposal_sigma2_tex.fetch<float>(x,y));
    float2 pt = make_float2((float)x, (float)y);
    float3 pt_w = cam1.cam2world(pt);
    float3 pt_h = make_float3(pt.x, pt.y, 1.f);
    float3 f_p = ::normalize(pt_w);

    float2 pt_mu = mu>1e-3f ? cam2.world2cam(T_mov_fix * (f_p*mu)) : pt;
    d_p_mu[y*stride_32fC2+x] = {pt_mu.x, pt_mu.y};

    float3 epi_line = F_mov_fix*pt_h;
    // epi_line=(a,b,c) -> line equation: ax+by+c=0 -> y=(-c-ax)/b -> k=-a/b
    float2 epi_line_slope = make_float2(1.0f, -epi_line.x/epi_line.y);
    float2 epi_vec = ::normalize(epi_line_slope);
    d_epi_vec[y*stride_32fC2+x] = {epi_vec.x, epi_vec.y};

    float u0 = u0_tex.fetch<float>(x,y);
    float2 px_u0 = pt_mu + epi_vec*u0; // assuming that epi_vec is the unit vec
    float2 px_u0_p = px_u0 + 0.5f*epi_vec;
    float2 px_u0_m = px_u0 - 0.5f*epi_vec;

    float bd = .5f;
    // check if current mean projects in image /*and mark if not*/
    // and if warped point is within a certain image area
    if (
        (pt_mu.x > width-bd-1) || (pt_mu.y > height-bd-1) || (pt_mu.x < bd) || (pt_mu.y < bd) ||
        (px_u0.x < bd) || (x < bd) || (px_u0.y > width-bd-1) || (x > width-bd-1) ||
        (px_u0.y < bd) || (px_u0.y > height-bd-1) || (y < bd) || (y > height-bd-1))
    {
      ix[c] = 0.0f;
      it[c] = 0.0f;
    }
    else
    {
      Pixel i1_c, i2_w_c, i2_w_m, i2_w_p;

      i1_tex.fetch(i1_c, x, y);

      i2_tex.fetch(i2_w_c, px_u0.x, px_u0.y);
      i2_tex.fetch(i2_w_m, px_u0_m.x, px_u0_m.y);
      i2_tex.fetch(i2_w_p,px_u0_p.x, px_u0_p.y);

      // spatial gradient on warped image
      ix[c] = i2_w_p - i2_w_m;
      // temporal gradient between the warped moving image and the fixed image
      it[c] = i2_w_c - i1_c;
    }
  }
}


} // namespace cu
} // namespace imp



#endif // IMP_CU_K_WARPED_GRADIENTS_CUH

