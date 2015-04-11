#ifndef IMP_CU_K_EPIPOLAR_STEREO_PRECOND_HUBER_L1_CUH
#define IMP_CU_K_EPIPOLAR_STEREO_PRECOND_HUBER_L1_CUH

#include <cuda_runtime_api.h>
#include <imp/core/types.hpp>
#include <imp/core/pixel.hpp>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_k_derivative_epipolar.cuh>
#include <imp/cuda_toolkit/helper_math.h>


namespace imp {
namespace cu {


//-----------------------------------------------------------------------------
template<typename Pixel>
__global__ void k_preconditioner(Pixel* xi, size_type stride,
                                 std::uint32_t width, std::uint32_t height,
                                 // std::uint32_t roi_x, std::uint32_t roi_y,
                                 float lambda, Texture2D ix_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    Pixel ix;
    ix_tex.fetch(ix, x, y);
    xi[y*stride+x] = 4 + fabs(lambda) * fabs(ix);
  }
}

//-----------------------------------------------------------------------------
/** restricts the udpate to +/- lin_step around the given value in lin_tex
 * @note \a d_srcdst and the return value is identical.
 * @todo (MWE) move function to common kernel def file for all stereo models
 */
template<typename Pixel>
__device__ Pixel k_linearized_update(Pixel& d_srcdst, Texture2D& lin_tex,
                                     const float lin_step,
                                     const int x, const int y)
{
  Pixel lin = lin_tex.fetch<Pixel>(x, y);
  d_srcdst = max(lin-lin_step,
                 min(lin+lin_step, d_srcdst));
  return d_srcdst;
}

//-----------------------------------------------------------------------------
/**
 * @brief k_primalUpdate is the Huber-L1-Precondition model's primal update kernel
 * @note PPixel and DPixel denote for the Pixel type/dimension of primal and dual variable
 */
template<typename PPixel>
__global__ void k_epiPrimalUpdate(PPixel* d_u, PPixel* d_u_prev, const size_type stride,
                                  std::uint32_t width, std::uint32_t height,
                                  Vec32fC2 epipole,
                                  const float lambda, const float tau,
                                  const float lin_step,
                                  Texture2D u_tex, Texture2D u0_tex,
                                  Texture2D pu_tex, Texture2D q_tex,
                                  Texture2D ix_tex, Texture2D xi_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<width && y<height)
  {
    float u_prev = u_tex.fetch<float>(x, y);
    float q = q_tex.fetch<float>(x, y);
    float ix = ix_tex.fetch<float>(x, y);
    float xi = xi_tex.fetch<float>(x, y);

    float2 epole = make_float2(epipole.x, epipole.y);
    float2 xc = make_float2((float)x, (float)y);
    float2 evec = epole-xc;
    float2 epi_vec = ::normalize(evec);

    float div = dpAdEpi(pu_tex, epi_vec, x, y, width, height);

    float u = u_prev - tau/xi * (-div + lambda*ix*q);

    u = k_linearized_update(u, u0_tex, lin_step, x, y);
    d_u[y*stride+x] = u;
    d_u_prev[y*stride+x] = 2.f*u - u_prev;
  }
}

////-----------------------------------------------------------------------------
///**
// * @brief k_primalUpdate is the Huber-L1-Precondition model's primal update kernel
// * @note PPixel and DPixel denote for the Pixel type/dimension of primal and dual variable
// */
//template<typename PPixel>
//__global__ void k_primalUpdate(PPixel* d_u, PPixel* d_u_prev, const size_type stride,
//                               std::uint32_t width, std::uint32_t height,
//                               const Vec32fC2 epipole,
//                               const float tau, const float lin_step,
//                               Texture2D lambda_tex,
//                               Texture2D u_tex, Texture2D u0_tex,
//                               Texture2D pu_tex, Texture2D q_tex,
//                               Texture2D ix_tex, Texture2D xi_tex)
//{
//  const int x = blockIdx.x*blockDim.x + threadIdx.x;
//  const int y = blockIdx.y*blockDim.y + threadIdx.y;

//  if (x<width && y<height)
//  {
//    float u_prev = u_tex.fetch<float>(x, y);
//    float q = q_tex.fetch<float>(x, y);
//    float ix = ix_tex.fetch<float>(x, y);
//    float xi = xi_tex.fetch<float>(x, y);

//    float div = dpAd(pu_tex, x, y, epipole, width, height);

//    float lambda = lambda_tex.fetch<float>(x,y);
//    float u = u_prev - tau/xi * (-div + lambda*ix*q);

//    u = k_linearized_update(u, u0_tex, lin_step, x, y);
//    d_u[y*stride+x] = u;
//    d_u_prev[y*stride+x] = 2.f*u - u_prev;
//  }
//}

//-----------------------------------------------------------------------------
template<typename PPixel, typename DPixel>
__global__ void k_epiDualUpdate(DPixel* d_pu, const size_type stride_pu,
                                PPixel* d_q, const size_type stride_q,
                                std::uint32_t width, std::uint32_t height,
                                Vec32fC2 epipole,
                                const float lambda, const float eps_u,
                                const float sigma, const float eta,
                                Texture2D u_prev_tex, Texture2D u0_tex,
                                Texture2D pu_tex, Texture2D q_tex,
                                Texture2D ix_tex, Texture2D it_tex)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x<width && y<height)
  {
    const float sigma_by_eta = sigma/eta;

    float2 epole = make_float2(epipole.x, epipole.y);
    float2 xc = make_float2((float)x, (float)y);
    float2 evec = epole-xc;
    float2 epi_vec = ::normalize(evec);


    // update pu
    float2 du = dpEpi(u_prev_tex, epi_vec, x, y);
    float2 pu = pu_tex.fetch<float2>(x,y);
    pu  = (pu + sigma_by_eta*du) / (1.f + sigma_by_eta*eps_u);
    pu = pu / max(1.0f, length(pu));
    d_pu[y*stride_pu+x] = {pu.x, pu.y};

    // update q
    float u_prev = u_prev_tex.fetch<float>(x, y);
    float u0 = u0_tex.fetch<float>(x, y);
    float q = q_tex.fetch<float>(x, y);
    float ix = ix_tex.fetch<float>(x, y);
    float it = it_tex.fetch<float>(x, y);
    const float sigma_q = sigma / max(1e-6f, lambda * fabs(ix));
    q = q + lambda*sigma_q * (it + ix*(u_prev-u0));
    d_q[y*stride_q+x] = max(-1.f, min(1.f, q));
  }
}

////-----------------------------------------------------------------------------
//template<typename PPixel, typename DPixel>
//__global__ void k_dualUpdate(DPixel* d_pu, const size_type stride_pu,
//                             PPixel* d_q, const size_type stride_q,
//                             std::uint32_t width, std::uint32_t height,
//                             const float eps_u,
//                             const float sigma, const float eta,
//                             Texture2D lambda_tex,
//                             Texture2D u_prev_tex, Texture2D u0_tex,
//                             Texture2D pu_tex, Texture2D q_tex,
//                             Texture2D ix_tex, Texture2D it_tex)
//{
//  const int x = blockIdx.x*blockDim.x + threadIdx.x;
//  const int y = blockIdx.y*blockDim.y + threadIdx.y;
//  if (x<width && y<height)
//  {
//    const float sigma_by_eta = sigma/eta;

//    // update pu
//    float2 du = dp(u_prev_tex, x, y);
//    float2 pu = pu_tex.fetch<float2>(s);
//    pu  = (pu + sigma_by_eta*du) / (1.f + sigma_by_eta*eps_u);
//    pu = pu / max(1.0f, length(pu));
//    d_pu[y*stride_pu+x] = {pu.x, pu.y};

//    // update q
//    float u_prev = u_prev_tex.fetch<float>(x, y);
//    float u0 = u0_tex.fetch<float>(x, y);
//    float q = q_tex.fetch<float>(x, y);
//    float ix = ix_tex.fetch<float>(x, y);
//    float it = it_tex.fetch<float>(x, y);
//    float lambda = lambda_tex.fetch<float>(x,y);
//    const float sigma_q = sigma / max(1e-6f, lambda * fabs(ix));
//    q = q + lambda*sigma_q * (it + ix*(u_prev-u0));
//    d_q[y*stride_q+x] = max(-1.f, min(1.f, q));
//  }
//}


} // namespace cu
} // namespace imp



#endif // IMP_CU_K_EPIPOLAR_STEREO_PRECOND_HUBER_L1_CUH

