#ifndef IMP_CU_EPIPOLAR_STEREO_PRECOND_HUBER_L1_CUH
#define IMP_CU_EPIPOLAR_STEREO_PRECOND_HUBER_L1_CUH

#include <cstdint>
#include <memory>

#include <imp/cudepth/solver_stereo_abstract.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cucore/cu_matrix.cuh>
#include <imp/cucore/cu_se3.cuh>
#include <imp/cucore/cu_pinhole_camera.cuh>
#include <imp/core/size.hpp>

namespace imp {
namespace cu {

// forward decl
class VariationalStereoParameters;
class Texture2D;

/**
 * @brief The StereoCtFWarpingLevelPrecondHuberL1 class
 */
class SolverEpipolarStereoPrecondHuberL1 : public SolverStereoAbstract
{
public:
  SolverEpipolarStereoPrecondHuberL1() = delete;
  virtual ~SolverEpipolarStereoPrecondHuberL1();

  SolverEpipolarStereoPrecondHuberL1(const std::shared_ptr<Parameters>& params,
                                     imp::Size2u size, size_type level,
                                     const std::vector<cu::PinholeCamera>& cams,
                                     const cu::Matrix3f& F,
                                     const std::vector<Vec32fC2>& epipoles,
                                     const cu::SE3<float>& T_mov_fix,
                                     const DisparityImage& depth_proposal,
                                     const imp::cu::ImageGpu32fC1& depth_proposal_sigma2);

  virtual void init();
  virtual void init(const SolverStereoAbstract& rhs);
  virtual inline void setFundamentalMatrix(const cu::Matrix3f& F) {F_ = F;}

  virtual void solve(std::vector<ImagePtr> images);

  virtual inline ImagePtr getDisparities() {return u_;}

protected:
  DisparityImagePtr u_; //!< disparities (result)
  std::unique_ptr<Image> u_prev_; //!< disparities results from previous iteration
  std::unique_ptr<Image> u0_; //!< disparities results from previous warp
  std::unique_ptr<VectorImage> pu_; //!< dual variable for primal variable
  std::unique_ptr<Image> q_; //!< dual variable for data term
  std::unique_ptr<Image> ix_; //!< spatial gradients on moving (warped) image
  std::unique_ptr<Image> it_; //!< temporal gradients between warped and fixed image
  std::unique_ptr<Image> xi_; //!< preconditioner
  std::unique_ptr<Image> g_; //!< for edge weighting


  cu::Matrix3f F_;
  std::vector<Vec32fC2> epipoles_;
  std::vector<cu::PinholeCamera> cams_;
  cu::SE3<float> T_mov_fix_;
  std::unique_ptr<DisparityImage> mu_;
  std::unique_ptr<DisparityImage> sigma2_;
  imp::cu::ImageGpu32fC2::UPtr pt_mu_; //!< the projection of the current depth proposal in the moving image
  imp::cu::ImageGpu32fC2::UPtr epi_vec_; //!< the epipolar vector (direction; unit-length)


  // textures
  std::unique_ptr<Texture2D> lambda_tex_;
  std::unique_ptr<Texture2D> i1_tex_;
  std::unique_ptr<Texture2D> i2_tex_;
  std::unique_ptr<Texture2D> u_tex_;
  std::unique_ptr<Texture2D> u_prev_tex_;
  std::unique_ptr<Texture2D> u0_tex_;
  std::unique_ptr<Texture2D> pu_tex_;
  std::unique_ptr<Texture2D> q_tex_;
  std::unique_ptr<Texture2D> ix_tex_;
  std::unique_ptr<Texture2D> it_tex_;
  std::unique_ptr<Texture2D> xi_tex_;
  std::unique_ptr<Texture2D> g_tex_;

  std::unique_ptr<Texture2D> mu_tex_;
  std::unique_ptr<Texture2D> sigma2_tex_;
  std::unique_ptr<Texture2D> pt_mu_tex_;
  std::unique_ptr<Texture2D> epi_vec_tex_;
};

} // namespace cu
} // namespace imp

#endif // IMP_CU_EPIPOLAR_STEREO_PRECOND_HUBER_L1_CUH
