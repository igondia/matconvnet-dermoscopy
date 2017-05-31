// @file nnpooling.cu
// @brief Pooling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnweakloss.hpp"
#include "impl/weakloss.hpp"
#include "mexutils.h"
#if ENABLE_GPU
#include "datacu.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                nnaggregation_mask_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, type) \
status = vl::impl::weakloss<deviceType, type>::forward \
((type*)output.getMemory(), (type const*)data.getMemory(),  (type const*)pcoords.getMemory(),\
(type const*)labels.getMemory(), (type *)lambda.getMemory(),\
(type const*)A.getMemory(),(type const*)b.getMemory(),(type const*)beta.getMemory(), maxLambda,\
data.getHeight(), data.getWidth(), data.getDepth(), data.getSize());

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ;\
}




vl::ErrorCode
vl::nnweakloss_forward(vl::Context& context,
        			   vl::Tensor output,
        			   vl::Tensor data,
        			   vl::Tensor pcoords,
        			   vl::Tensor labels,
        			   vl::Tensor lambda,
        			   vl::Tensor A,
        			   vl::Tensor b,
        			   vl::Tensor beta,
        			   float maxLambda)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = output.getDeviceType() ;
  vl::DataType dataType = output.getDataType() ;
  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::VLDT_GPU:


      DISPATCH2(vl::VLDT_GPU) ;
      if (status == vl::VLE_Cuda) {
    	  context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }
  
  return context.passError(status, "nnweakloss_forward") ;
}

/* ---------------------------------------------------------------- */
/*                                               nnaggregation_mask_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#undef DISPATCH2

// backward max and average want slightly different argument lists

#define DISPATCH(deviceType, type) \
status = vl::impl::weakloss<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)data.getMemory(), (type const*)pcoords.getMemory(),\
(type const*)labels.getMemory(), (type*)lambda.getMemory(), (type const*)derOutput.getMemory(),\
(type const*)A.getMemory(),(type const*)b.getMemory(),(type const*)beta.getMemory(), maxLambda,\
derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize());

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnweakloss_backward(Context& context,
                       Tensor derData,
                       Tensor data,
		       		   Tensor pcoords,
                       Tensor labels,
                       Tensor lambda,
		       		   Tensor derOutput,
		       		   Tensor A,
		       		   Tensor b,
		       		   Tensor beta,
		       		   float maxLambda)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = derOutput.getDeviceType() ;
  vl::DataType dataType = derOutput.getDataType() ;
  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        /*
         Unfortunately CuDNN requires both the input and the output pooling arrays
         to be available for computing derivatives, whereas MatConvNet only requires the input one.
         */
      }
#endif

      DISPATCH2(vl::VLDT_GPU) ;
      if (status == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError("aggregation_*::backward")) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnaggregation_backward") ;
}
