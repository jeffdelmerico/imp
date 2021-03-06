#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/bridge/opencv/image_cv.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_imgproc/cu_rof_denoising.cuh>
#include <imp/bridge/opencv/cu_cv_bridge.hpp>

int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    // ROF denoising 8uC1
    {
      std::shared_ptr<imp::cu::ImageGpu8uC1> d1_lena_8uC1;
      imp::cu::cvBridgeLoad(d1_lena_8uC1, "/home/mwerlberger/data/std/Lena.tiff",
                            imp::PixelOrder::gray);
      std::shared_ptr<imp::cu::ImageGpu8uC1> d_lena_denoised_8uC1(
            new imp::cu::ImageGpu8uC1(*d1_lena_8uC1));

      imp::cu::RofDenoising8uC1 rof;
      std::cout << "\n" << rof << std::endl << std::endl;
      rof.denoise(d_lena_denoised_8uC1, d1_lena_8uC1);

      // show results
      imp::cu::cvBridgeShow("lena input 8u", *d1_lena_8uC1);
      imp::cu::cvBridgeShow("lena denoised 8u", *d_lena_denoised_8uC1);
    }

//    imp::cvBridgeSave("test.png", h_lena_denoised_8uC1);

    // ROF denoising 32fC1
    {
      std::shared_ptr<imp::cu::ImageGpu32fC1> d1_lena_32fC1;
      imp::cu::cvBridgeLoad(d1_lena_32fC1, "/home/mwerlberger/data/std/Lena.tiff",
                            imp::PixelOrder::gray);
      std::shared_ptr<imp::cu::ImageGpu32fC1> d_lena_denoised_32fC1(
            new imp::cu::ImageGpu32fC1(*d1_lena_32fC1));

      imp::cu::RofDenoising32fC1 rof_32fC1;
      rof_32fC1.denoise(d_lena_denoised_32fC1, d1_lena_32fC1);

      imp::cu::cvBridgeShow("lena input 32f", *d1_lena_32fC1);
      imp::cu::cvBridgeShow("lena denoised 32f", *d_lena_denoised_32fC1);
    }

    cv::waitKey();
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }

  return EXIT_SUCCESS;

}
