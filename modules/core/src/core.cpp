#include <imp/core.h>
//#include <imp/core/copy.h>
//#include <imp/core/setvalue.h>
//#include <imp/core/clamp.h>
//#include <imp/core/convert.h>

namespace imp {

/* ***************************************************************************
 * 1D COPY
 * ***************************************************************************/

/*
// 1D copy device -> device;
void copy(const LinearDeviceMemory_8u_C1* src, LinearDeviceMemory_8u_C1* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C2* src, LinearDeviceMemory_8u_C2* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C3* src, LinearDeviceMemory_8u_C3* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C4* src, LinearDeviceMemory_8u_C4* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C1* src, LinearDeviceMemory_16u_C1* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C2* src, LinearDeviceMemory_16u_C2* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C3* src, LinearDeviceMemory_16u_C3* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C4* src, LinearDeviceMemory_16u_C4* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32s_C1* src, LinearDeviceMemory_32s_C1* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32s_C2* src, LinearDeviceMemory_32s_C2* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32s_C3* src, LinearDeviceMemory_32s_C3* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32s_C4* src, LinearDeviceMemory_32s_C4* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C1* src, LinearDeviceMemory_32f_C1* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C2* src, LinearDeviceMemory_32f_C2* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C3* src, LinearDeviceMemory_32f_C3* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C4* src, LinearDeviceMemory_32f_C4* dst){ imp::copy(src,dst); }

// 1D copy host -> device;
void copy(const LinearHostMemory_8u_C1* src, LinearDeviceMemory_8u_C1* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_8u_C2* src, LinearDeviceMemory_8u_C2* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_8u_C3* src, LinearDeviceMemory_8u_C3* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_8u_C4* src, LinearDeviceMemory_8u_C4* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_16u_C1* src, LinearDeviceMemory_16u_C1* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_16u_C2* src, LinearDeviceMemory_16u_C2* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_16u_C3* src, LinearDeviceMemory_16u_C3* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_16u_C4* src, LinearDeviceMemory_16u_C4* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_32s_C1* src, LinearDeviceMemory_32s_C1* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_32s_C2* src, LinearDeviceMemory_32s_C2* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_32s_C3* src, LinearDeviceMemory_32s_C3* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_32s_C4* src, LinearDeviceMemory_32s_C4* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_32f_C1* src, LinearDeviceMemory_32f_C1* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_32f_C2* src, LinearDeviceMemory_32f_C2* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_32f_C3* src, LinearDeviceMemory_32f_C3* dst){ imp::copy(src,dst); }
void copy(const LinearHostMemory_32f_C4* src, LinearDeviceMemory_32f_C4* dst){ imp::copy(src,dst); }

// 1D copy device -> host;
void copy(const LinearDeviceMemory_8u_C1* src, LinearHostMemory_8u_C1* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C2* src, LinearHostMemory_8u_C2* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C3* src, LinearHostMemory_8u_C3* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C4* src, LinearHostMemory_8u_C4* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C1* src, LinearHostMemory_16u_C1* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C2* src, LinearHostMemory_16u_C2* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C3* src, LinearHostMemory_16u_C3* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C4* src, LinearHostMemory_16u_C4* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32s_C1* src, LinearHostMemory_32s_C1* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32s_C2* src, LinearHostMemory_32s_C2* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32s_C3* src, LinearHostMemory_32s_C3* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32s_C4* src, LinearHostMemory_32s_C4* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C1* src, LinearHostMemory_32f_C1* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C2* src, LinearHostMemory_32f_C2* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C3* src, LinearHostMemory_32f_C3* dst){ imp::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C4* src, LinearHostMemory_32f_C4* dst){ imp::copy(src,dst); }
*/

///* ***************************************************************************
// * 2D COPY
// * ***************************************************************************/

//// 2D copy host -> host;
//void copy(const ImageCpu_8u_C1* src, ImageCpu_8u_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_8u_C2* src, ImageCpu_8u_C2* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_8u_C3* src, ImageCpu_8u_C3* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_8u_C4* src, ImageCpu_8u_C4* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_32s_C1* src, ImageCpu_32s_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_32f_C1* src, ImageCpu_32f_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_32f_C2* src, ImageCpu_32f_C2* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_32f_C3* src, ImageCpu_32f_C3* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_32f_C4* src, ImageCpu_32f_C4* dst) { imp::copy(src, dst); }

//// 2D copy device -> device;
//void copy(const ImageGpu_8u_C1* src, ImageGpu_8u_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_8u_C2* src, ImageGpu_8u_C2* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_8u_C3* src, ImageGpu_8u_C3* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_8u_C4* src, ImageGpu_8u_C4* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_32s_C1* src, ImageGpu_32s_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_32f_C2* src, ImageGpu_32f_C2* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_32f_C3* src, ImageGpu_32f_C3* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_32f_C4* src, ImageGpu_32f_C4* dst) { imp::copy(src, dst); }

//// 2D copy host -> device;
//void copy(const ImageCpu_8u_C1* src, ImageGpu_8u_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_8u_C2* src, ImageGpu_8u_C2* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_8u_C3* src, ImageGpu_8u_C3* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_8u_C4* src, ImageGpu_8u_C4* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_32s_C1* src, ImageGpu_32s_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_32f_C1* src, ImageGpu_32f_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_32f_C2* src, ImageGpu_32f_C2* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_32f_C3* src, ImageGpu_32f_C3* dst) { imp::copy(src, dst); }
//void copy(const ImageCpu_32f_C4* src, ImageGpu_32f_C4* dst) { imp::copy(src, dst); }

//// 2D copy device -> host;
//void copy(const ImageGpu_8u_C1* src, ImageCpu_8u_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_8u_C2* src, ImageCpu_8u_C2* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_8u_C3* src, ImageCpu_8u_C3* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_8u_C4* src, ImageCpu_8u_C4* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_32s_C1* src, ImageCpu_32s_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_32f_C1* src, ImageCpu_32f_C1* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_32f_C2* src, ImageCpu_32f_C2* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_32f_C3* src, ImageCpu_32f_C3* dst) { imp::copy(src, dst); }
//void copy(const ImageGpu_32f_C4* src, ImageCpu_32f_C4* dst) { imp::copy(src, dst); }


///* ***************************************************************************
// * 3D COPY
// * ***************************************************************************/

//// 3D copy host -> host;
//void copy(const VolumeCpu_8u_C1* src, VolumeCpu_8u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_8u_C2* src, VolumeCpu_8u_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_8u_C4* src, VolumeCpu_8u_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_16u_C1* src, VolumeCpu_16u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32f_C1* src, VolumeCpu_32f_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32f_C2* src, VolumeCpu_32f_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32f_C3* src, VolumeCpu_32f_C3* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32f_C4* src, VolumeCpu_32f_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32u_C1* src, VolumeCpu_32u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32u_C2* src, VolumeCpu_32u_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32u_C4* src, VolumeCpu_32u_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32s_C1* src, VolumeCpu_32s_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32s_C2* src, VolumeCpu_32s_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32s_C4* src, VolumeCpu_32s_C4* dst) { imp::copy(src, dst); }



//// 3D copy device -> device;
//void copy(const VolumeGpu_8u_C1* src, VolumeGpu_8u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_8u_C2* src, VolumeGpu_8u_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_8u_C4* src, VolumeGpu_8u_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_16u_C1* src, VolumeGpu_16u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32f_C1* src, VolumeGpu_32f_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32f_C2* src, VolumeGpu_32f_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32f_C4* src, VolumeGpu_32f_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32u_C1* src, VolumeGpu_32u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32u_C2* src, VolumeGpu_32u_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32u_C4* src, VolumeGpu_32u_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32s_C1* src, VolumeGpu_32s_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32s_C2* src, VolumeGpu_32s_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32s_C4* src, VolumeGpu_32s_C4* dst) { imp::copy(src, dst); }


//// 3D copy host -> device;
//void copy(const VolumeCpu_8u_C1* src, VolumeGpu_8u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_8u_C2* src, VolumeGpu_8u_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_8u_C4* src, VolumeGpu_8u_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_16u_C1* src, VolumeGpu_16u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32f_C1* src, VolumeGpu_32f_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32f_C2* src, VolumeGpu_32f_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32f_C3* src, VolumeGpu_32f_C3* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32f_C4* src, VolumeGpu_32f_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32u_C1* src, VolumeGpu_32u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32u_C2* src, VolumeGpu_32u_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32u_C4* src, VolumeGpu_32u_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32s_C1* src, VolumeGpu_32s_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32s_C2* src, VolumeGpu_32s_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeCpu_32s_C4* src, VolumeGpu_32s_C4* dst) { imp::copy(src, dst); }


//// 3D copy device -> host;
//void copy(const VolumeGpu_8u_C1* src, VolumeCpu_8u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_8u_C2* src, VolumeCpu_8u_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_8u_C4* src, VolumeCpu_8u_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_16u_C1* src, VolumeCpu_16u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32f_C1* src, VolumeCpu_32f_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32f_C2* src, VolumeCpu_32f_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32f_C3* src, VolumeCpu_32f_C3* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32f_C4* src, VolumeCpu_32f_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32u_C1* src, VolumeCpu_32u_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32u_C2* src, VolumeCpu_32u_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32u_C4* src, VolumeCpu_32u_C4* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32s_C1* src, VolumeCpu_32s_C1* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32s_C2* src, VolumeCpu_32s_C2* dst) { imp::copy(src, dst); }
//void copy(const VolumeGpu_32s_C4* src, VolumeCpu_32s_C4* dst) { imp::copy(src, dst); }



/* ***************************************************************************
     SET
 * ***************************************************************************/

//void setValue(const unsigned char& value, LinearDeviceMemory_8u_C1* srcdst)
//{imp::setValue(value, srcdst);}
//void setValue(const int& value, LinearDeviceMemory_32s_C1* srcdst)
//{imp::setValue(value, srcdst);}
//void setValue(const float& value, LinearDeviceMemory_32f_C1* srcdst)
//{imp::setValue(value, srcdst);}

//void setValue(const unsigned char &value, ImageCpu_8u_C1* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uchar2 &value, ImageCpu_8u_C2* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uchar3 &value, ImageCpu_8u_C3* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uchar4 &value, ImageCpu_8u_C4* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const int &value, ImageCpu_32s_C1* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float &value, ImageCpu_32f_C1* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float2 &value, ImageCpu_32f_C2* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float3 &value, ImageCpu_32f_C3* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float4 &value, ImageCpu_32f_C4* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}

//void setValue(const unsigned char &value, ImageGpu_8u_C1* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uchar2 &value, ImageGpu_8u_C2* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uchar3 &value, ImageGpu_8u_C3* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uchar4 &value, ImageGpu_8u_C4* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const int &value, ImageGpu_32s_C1* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float &value, ImageGpu_32f_C1* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float2 &value, ImageGpu_32f_C2* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float3 &value, ImageGpu_32f_C3* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float4 &value, ImageGpu_32f_C4* srcdst, const IuRect& roi) {imp::setValue(value, srcdst, roi);}

//void setValue(const unsigned char &value, VolumeCpu_8u_C1* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uchar2 &value, VolumeCpu_8u_C2* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uchar4 &value, VolumeCpu_8u_C4* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float &value, VolumeCpu_32f_C1* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float2 &value, VolumeCpu_32f_C2* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float4 &value, VolumeCpu_32f_C4* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const unsigned int &value, VolumeCpu_32u_C1* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uint2 &value, VolumeCpu_32u_C2* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uint4 &value, VolumeCpu_32u_C4* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const int &value, VolumeCpu_32s_C1* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const int2 &value, VolumeCpu_32s_C2* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const int4 &value, VolumeCpu_32s_C4* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}



//void setValue(const unsigned char &value, VolumeGpu_8u_C1* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uchar2 &value, VolumeGpu_8u_C2* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uchar4 &value, VolumeGpu_8u_C4* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const unsigned short &value, VolumeGpu_16u_C1* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float &value, VolumeGpu_32f_C1* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float2 &value, VolumeGpu_32f_C2* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const float4 &value, VolumeGpu_32f_C4* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const unsigned int &value, VolumeGpu_32u_C1* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uint2 &value, VolumeGpu_32u_C2* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const uint4 &value, VolumeGpu_32u_C4* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const int &value, VolumeGpu_32s_C1* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const int2 &value, VolumeGpu_32s_C2* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}
//void setValue(const int4 &value, VolumeGpu_32s_C4* srcdst, const IuCube& roi) {imp::setValue(value, srcdst, roi);}


///* ***************************************************************************
//     CLAMP
// * ***************************************************************************/

//void clamp(const float& min, const float& max, iu::ImageGpu_32f_C1 *srcdst, const IuRect &roi)
//{ imp::clamp(min, max, srcdst, roi); }


///* ***************************************************************************
// *  MEMORY CONVERSIONS
// * ***************************************************************************/

//// conversion; device; 32-bit 3-channel -> 32-bit 4-channel
//void convert(const ImageGpu_32f_C3* src, const IuRect& src_roi, ImageGpu_32f_C4* dst, const IuRect& dst_roi)
//{imp::convert(src, src_roi, dst, dst_roi);}
//// conversion; device; 32-bit 4-channel -> 32-bit 3-channel
//void convert(const ImageGpu_32f_C4* src, const IuRect& src_roi, ImageGpu_32f_C3* dst, const IuRect& dst_roi)
//{imp::convert(src, src_roi, dst, dst_roi);}

//// [host] 2D bit depth conversion; 32f_C1 -> 8u_C1;
//void convert_32f8u_C1(const iu::ImageCpu_32f_C1* src, iu::ImageCpu_8u_C1* dst,
//                       float mul_constant, float add_constant)
//{imp::convert_32f8u_C1(src, dst, mul_constant, add_constant);}

//// [host] 2D bit depth conversion; 16u_C1 -> 32f_C1;
//void convert_16u32f_C1(const iu::ImageCpu_16u_C1* src, iu::ImageCpu_32f_C1 *dst,
//                       float mul_constant, float add_constant)
//{imp::convert_16u32f_C1(src, dst, mul_constant, add_constant);}

//// [device] 2D bit depth conversion: 32f_C1 -> 8u_C1
//void convert_32f8u_C1(const iu::ImageGpu_32f_C1* src, const IuRect& src_roi, iu::ImageGpu_8u_C1* dst, const IuRect& dst_roi,
//                     float mul_constant, unsigned char add_constant)
//{imp::convert_32f8u_C1(src, src_roi, dst, dst_roi, mul_constant, add_constant);}

//// [device] 2D bit depth conversion: 32f_C4 -> 8u_C4
//void convert_32f8u_C4(const iu::ImageGpu_32f_C4* src, const IuRect& src_roi, iu::ImageGpu_8u_C4* dst, const IuRect& dst_roi,
//                     float mul_constant, unsigned char add_constant)
//{imp::convert_32f8u_C4(src, src_roi, dst, dst_roi, mul_constant, add_constant);}


//// [device] 2D bit depth conversion: 8u_C1 -> 32f_C1
//void convert_8u32f_C1(const iu::ImageGpu_8u_C1* src, const IuRect& src_roi, iu::ImageGpu_32f_C1* dst, const IuRect& dst_roi,
//                     float mul_constant, float add_constant)
//{imp::convert_8u32f_C1(src, src_roi, dst, dst_roi, mul_constant, add_constant);}

//void convert_8u32f_C3C4(const iu::ImageGpu_8u_C3* src, const IuRect& src_roi, iu::ImageGpu_32f_C4* dst, const IuRect& dst_roi,
//                                float mul_constant, float add_constant)
//{imp::convert_8u32f_C3C4(src, src_roi, dst, dst_roi, mul_constant, add_constant);}


//// [device] 2D Color conversion from RGB to HSV (32-bit 4-channel)
//void convert_RgbHsv(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool normalize)
//{imp::convertRgbHsv(src, dst, normalize);}

//// [device] 2D Color conversion from HSV to RGB (32-bit 4-channel)
//void convert_HsvRgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool denormalize)
//{imp::convertHsvRgb(src, dst, denormalize);}


//// [device] 2D Color conversion from RGB to CIELAB (32-bit 4-channel)
//void convert_RgbLab(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool isNormalized)
//{ imp::convertRgbLab(src, dst, isNormalized); }


//// [device] 2D Color conversion from CIELAB to RGB (32-bit 4-channel)
//void convert_LabRgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst)
//{ imp::convertLabRgb(src, dst); }


} // namespace imp
