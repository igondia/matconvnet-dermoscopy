// @file nnpooling.cu
// @brief Pooling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nncircpooling_mask.hpp"
#include "impl/circpooling_mask.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                nncircpooling_mask_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, op, type) \
status = vl::impl::op<deviceType, type>::forward \
((type*)output.getMemory(), (type const*)data.getMemory(),  (type const*)pcoords.getMemory(),\
data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(), data.getDepth(), \
poolRings, poolAngles, \
overlapRing, overlapAngle, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, op, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, op, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCH3(deviceType) \
switch (method) { \
case vlCircPoolingMaskAverage : DISPATCH2(deviceType, circpooling_mask_average) ; break ; \
case vlCircPoolingMaskMax : DISPATCH2(deviceType, circpooling_mask_max) ; break ; \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCHCUDNN(dataType) \
status = vl::impl::nnpooling_cudnn<dataType>::forward \
(context, output, data, \
method, \
poolRings, poolAngles, \
overlapRing, overlapAngle, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCHCUDNN2() \
switch (dataType) { \
case VLDT_Float : DISPATCHCUDNN(VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCHCUDNN(VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nncircpooling_mask_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
		      		  vl::Tensor pcoords,
                      PoolingMethod method,
                      int poolRings, int poolAngles,
					  float overlapRing, float overlapAngle, \
                      int padTop, int padBottom,
                      int padLeft, int padRight)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = output.getDeviceType() ;
  vl::DataType dataType = output.getDataType() ;
  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH3(vl::VLDT_CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH3(VLDT_GPU) ;
      if (status == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nncircpooling_mask_forward") ;
}

/* ---------------------------------------------------------------- */
/*                                               nncircpooling_mask_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#undef DISPATCH2

// backward max and average want slightly differet argument lists

#define DISPATCH_circpooling_mask_average(deviceType, type) \
status = vl::impl::circpooling_mask_average<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)pcoords.getMemory(), (type const*)derOutput.getMemory(), \
derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(), derData.getDepth(), \
poolRings, poolAngles, \
overlapRing, overlapAngle, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCH_circpooling_mask_max(deviceType, type) \
status = vl::impl::circpooling_mask_max<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)data.getMemory(), (type const*)pcoords.getMemory(), (type const*)derOutput.getMemory(), \
derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(), derData.getDepth(),\
poolRings, poolAngles, \
overlapRing, overlapAngle, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case VLDT_Float : DISPATCH_ ## op (deviceType, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH_ ## op (deviceType, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nncircpooling_mask_backward(Context& context,
                       Tensor derData,
                       Tensor data,
		       		   Tensor pcoords,
                       Tensor derOutput,
                       PoolingMethod method,
                       int poolRings, int poolAngles,
					   float overlapRing, float overlapAngle,
                       int padTop, int padBottom,
                       int padLeft, int padRight)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = derOutput.getDeviceType() ;
  vl::DataType dataType = derOutput.getDataType() ;
  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH3(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH3(vl::VLDT_GPU) ;
      if (status == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError("circpooling_mask_*::backward")) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nncircpooling_mask_backward") ;
}
