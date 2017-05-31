// @file nnpooling.hpp
// @brief Pooling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnweakloss__
#define __vl__nnweakloss__

#include "data.hpp"
#include <stdio.h>

namespace vl {


  vl::ErrorCode
  nnweakloss_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor data,
                    vl::Tensor pcoords,
                    vl::Tensor labels,
                    vl::Tensor lambda,
                    vl::Tensor A,
                    vl::Tensor b,
                    vl::Tensor beta,
                    float maxLambda) ;

  vl::ErrorCode
  nnweakloss_backward(vl::Context& context,
                     vl::Tensor derData,
                     vl::Tensor data,
                     vl::Tensor pcoords,
                     vl::Tensor labels,
                     vl::Tensor lambda,
                     vl::Tensor derOutput,
                     vl::Tensor A,
                     vl::Tensor b,
                     vl::Tensor beta,
                     float maxLambda) ;
}


#endif /* defined(__vl__nnweakloss__) */
