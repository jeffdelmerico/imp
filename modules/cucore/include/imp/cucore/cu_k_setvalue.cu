#ifndef IMP_CU_K_SETVALUE_CU
#define IMP_CU_K_SETVALUE_CU

#include <stdio.h>

#include <cuda_runtime_api.h>

#if 0
#include <imp/cucore/cu_gpu_data.cuh>
#endif

namespace imp { namespace cu {

//-----------------------------------------------------------------------------
/**
 * @brief
 */
#if 0
template<class Pixel>
__global__ void k_setValue(GpuData2D<Pixel>* dst, const Pixel& value)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  // account for roi
  x+=dst->roi.x();
  y+=dst->roi.y();

  if (dst->inRoi(x,y))
  {
    dst->data[y*dst->stride+x] = value;
  }
}
#endif

template<class Pixel>
__global__ void k_setValue(Pixel* dst, size_t stride, const Pixel& value,
                           size_t width, size_t height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  printf("x=%d, y=%d, w=%d, h=%d, s=%d, /*dst=%p,*/ value=%d, c=%d\n",
         x, y, width, height, stride, /*dst,*/ value.c[0], y*stride+x);

  if (x>=0 && y>=0 && x<width && y<height)
  {
    dst[y*stride+x] = value;
  }
}


} // namespace cu
} // namespace imp

#endif // IMP_CU_K_SETVALUE_CU