#include <imp/cudepth/solver_epipolar_stereo_precond_huber_l1.cuh>

#include <cuda_runtime.h>

#include <glog/logging.h>

#include <imp/cudepth/variational_stereo_parameters.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cuimgproc/cu_image_filter.cuh>
#include <imp/cuimgproc/cu_image_transform.cuh>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_texture.cuh>
#include <imp/cucore/cu_math.cuh>
#include <imp/cucore/cu_k_setvalue.cuh>
#include <imp/cuimgproc/edge_detectors.cuh>

#include "cu_k_warped_gradients.cuh"
//#include "cu_k_stereo_ctf_warping_level_precond_huber_l1.cuh"
#include "cu_k_solver_epipolar_stereo_precond_huber_l1.cuh"
#include "cu_k_stereo_ctf_warping_level_precond_huber_l1_weighted.cuh"

//#include "k_epipolar_stereo_precond_huber_l1.cu"

#define USE_EDGES 0

namespace imp {
namespace cu {

//------------------------------------------------------------------------------
SolverEpipolarStereoPrecondHuberL1::~SolverEpipolarStereoPrecondHuberL1()
{
  // thanks to smart pointers
}

//------------------------------------------------------------------------------
SolverEpipolarStereoPrecondHuberL1::SolverEpipolarStereoPrecondHuberL1(
    const std::shared_ptr<Parameters>& params, imp::Size2u size, size_type level,
    const std::vector<cu::PinholeCamera>& cams,
    const cu::Matrix3f& F,
    const std::vector<Vec32fC2>& epipoles,
    const cu::SE3<float>& T_mov_fix,
    const imp::cu::ImageGpu32fC1& depth_proposal,
    const imp::cu::ImageGpu32fC1& depth_proposal_sigma2)
  : SolverStereoAbstract(params, size, level)
  , F_(F)
  , epipoles_(epipoles)
  , T_mov_fix_(T_mov_fix)

{
  u_.reset(new DisparityImage(size));
  u_prev_.reset(new Image(size));
  u0_.reset(new Image(size));
  pu_.reset(new VectorImage(size));
  q_.reset(new Image(size));
  ix_.reset(new Image(size));
  it_.reset(new Image(size));
  xi_.reset(new Image(size));
  g_.reset(new Image(size));

  mu_.reset(new DisparityImage(size));
  sigma2_.reset(new DisparityImage(size));
  pt_mu_.reset(new imp::cu::ImageGpu32fC2(size));
  epi_vec_.reset(new imp::cu::ImageGpu32fC2(size));

  float scale_factor = std::pow(params->ctf.scale_factor, level);

  if (depth_proposal.size() == size)
  {
    LOG(INFO) << "Copy depth proposals " << depth_proposal.size() << " to level0 "
              << mu_->size();
    depth_proposal.copyTo(*mu_);
    depth_proposal_sigma2.copyTo(*sigma2_);
  }
  else
  {
    float downscale_factor = 0.5f*((float)size.width()/(float)depth_proposal.width()+
                                   (float)size.height()/(float)depth_proposal.height());

    if (params_->verbose>2)
      LOG(INFO) << "depth proposal downscaled to level: " << level << "; size: " << size
                << "; downscale_factor: " << downscale_factor;

    imp::cu::resample(*mu_, depth_proposal);
    imp::cu::resample(*sigma2_, depth_proposal_sigma2);
  }

  // assuming we receive the camera matrix for level0
  if  (level == 0)
  {
    cams_ = cams;
  }
  else
  {
    for (auto cam : cams)
    {
      cu::PinholeCamera scaled_cam = cam * scale_factor;
      cams_.push_back(scaled_cam);
    }
  }
}

//------------------------------------------------------------------------------
void SolverEpipolarStereoPrecondHuberL1::init()
{
  u_->setValue(0.0f);
  pu_->setValue(0.0f);
  q_->setValue(0.0f);
  // other variables are init and/or set when needed!
}

//------------------------------------------------------------------------------
void SolverEpipolarStereoPrecondHuberL1::init(const SolverStereoAbstract& rhs)
{
  const SolverEpipolarStereoPrecondHuberL1* from =
      dynamic_cast<const SolverEpipolarStereoPrecondHuberL1*>(&rhs);

  float inv_sf = 1./params_->ctf.scale_factor; // >1 for adapting prolongated disparities

  if(params_->ctf.apply_median_filter)
  {
    imp::cu::filterMedian3x3(*from->u0_, *from->u_);
    imp::cu::resample(*u_, *from->u0_, imp::InterpolationMode::point, false);
  }
  else
  {
    imp::cu::resample(*u_, *from->u_, imp::InterpolationMode::point, false);
  }
  *u_ *= inv_sf;

  imp::cu::resample(*pu_, *from->pu_, imp::InterpolationMode::point, false);
  imp::cu::resample(*q_, *from->q_, imp::InterpolationMode::point, false);
}

//------------------------------------------------------------------------------
void SolverEpipolarStereoPrecondHuberL1::solve(std::vector<ImagePtr> images)
{
  if (params_->verbose > 0)
    std::cout << "SolverEpipolarStereoPrecondHuberL1: solving level " << level_ << " with " << images.size() << " images" << std::endl;

  // sanity check:
  // TODO

  // constants
  constexpr float tau = 0.95f;
  constexpr float sigma = 0.95f;
  float lin_step = 0.5f;
  Fragmentation<16,16> frag(size_);
  constexpr float eta = 2.0f;

  // init
  u_->copyTo(*u_prev_);


  // check if a pointwise lambda is set in the parameters. otherwise we create
  // a local one to simplify kernel interfaces
  cu::ImageGpu32fC1::Ptr lambda;
  if (params_->lambda_pointwise)
    lambda = params_->lambda_pointwise;
  else
  {
    // make it as small as possible to reduce memory overhead. access is then
    // handled by the texture
    lambda.reset(new ImageGpu32fC1(1,1));
    lambda->setValue(params_->lambda);
  }
  lambda_tex_ = lambda->genTexture(false,cudaFilterModePoint,
                                   cudaAddressModeClamp, cudaReadModeElementType);

  // textures
  i1_tex_ = images.at(0)->genTexture(false, cudaFilterModeLinear);
  i2_tex_ = images.at(1)->genTexture(false, cudaFilterModeLinear);
  u_tex_ = u_->genTexture(false, cudaFilterModeLinear);
  u_prev_tex_ =  u_prev_->genTexture(false, cudaFilterModeLinear);
  u0_tex_ =  u0_->genTexture(false, cudaFilterModeLinear);
  pu_tex_ =  pu_->genTexture(false, cudaFilterModeLinear);
  q_tex_ =  q_->genTexture(false, cudaFilterModeLinear);
  ix_tex_ =  ix_->genTexture(false, cudaFilterModeLinear);
  it_tex_ =  it_->genTexture(false, cudaFilterModeLinear);
  xi_tex_ =  xi_->genTexture(false, cudaFilterModeLinear);
  g_tex_ =  g_->genTexture(false, cudaFilterModeLinear);
  mu_tex_ =  mu_->genTexture(false, cudaFilterModeLinear);
  sigma2_tex_ =  sigma2_->genTexture(false, cudaFilterModeLinear);
  pt_mu_tex_ =  pt_mu_->genTexture(false, cudaFilterModeLinear);
  epi_vec_tex_ =  epi_vec_->genTexture(false, cudaFilterModeLinear);



  // compute edge weight
  imp::cu::naturalEdges(*g_, *images.at(0),
                        params_->edge_sigma, params_->edge_alpha, params_->edge_q);



  // warping
  for (std::uint32_t warp = 0; warp < params_->ctf.warps; ++warp)
  {
    if (params_->verbose > 5)
      std::cout << "SOLVING warp iteration of Huber-L1 stereo model." << std::endl;

    u_->copyTo(*u0_);


    // compute warped spatial and temporal gradients
    k_warpedGradientsEpipolarConstraint
        <<<
          frag.dimGrid, frag.dimBlock
        >>> (ix_->data(), it_->data(), ix_->stride(),
             //todo mas
             pt_mu_->data(), epi_vec_->data(), pt_mu_->stride(),
             size_.width(), size_.height(),
             cams_.at(0), cams_.at(1), F_, T_mov_fix_,
             *i1_tex_, *i2_tex_, *u0_tex_,
             *mu_tex_);

    // compute preconditioner
    //! @todo  (MWE) fix preconditioner with pointwise lambda!
#if USE_EDGES
    k_preconditionerWeighted
        <<<
          frag.dimGrid, frag.dimBlock
        >>> (xi_->data(), xi_->stride(), xi_->width(), xi_->height(),
             params_->lambda, *ix_tex_, *g_tex_);
#else
    k_preconditioner
        <<<
          frag.dimGrid, frag.dimBlock
        >>> (xi_->data(), xi_->stride(), xi_->width(), xi_->height(),
             params_->lambda, *ix_tex_);
#endif

    for (std::uint32_t iter = 0; iter < params_->ctf.iters; ++iter)
    {
      // dual update kernel
      k_epiDualUpdate
          <<<
            frag.dimGrid, frag.dimBlock
          >>> (pu_->data(), pu_->stride(), q_->data(), q_->stride(),
               size_.width(), size_.height(),
               epipoles_.at(0),
               params_->lambda, params_->eps_u, sigma, eta, //*lambda_tex_,
               *u_prev_tex_, *u0_tex_, *pu_tex_, *q_tex_, *ix_tex_, *it_tex_);

      // and primal update kernel
#if USE_EDGES
      // and primal update kernel
      k_primalUpdateWeighted
          <<<
            frag.dimGrid, frag.dimBlock
          >>> (u_->data(), u_prev_->data(), u_->stride(),
               size_.width(), size_.height(),
               tau, lin_step, *lambda_tex_,
               *u_tex_, *u0_tex_, *pu_tex_, *q_tex_, *ix_tex_, *xi_tex_, *g_tex_);
#else
      k_epiPrimalUpdate
          <<<
            frag.dimGrid, frag.dimBlock
          >>> (u_->data(), u_prev_->data(), u_->stride(),
               size_.width(), size_.height(),
               epipoles_.at(0),
               params_->lambda, tau, lin_step, //*lambda_tex_,
               *u_tex_, *u0_tex_, *pu_tex_, *q_tex_, *ix_tex_, *xi_tex_);
#endif
    } // iters
    lin_step /= 1.2f;

  } // warps



  IMP_CUDA_CHECK();
}



} // namespace cu
} // namespace imp

