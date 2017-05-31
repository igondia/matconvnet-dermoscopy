// @file nnpooling.hpp
// @brief Pooling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnaggregation__
#define __vl__nnaggregation__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  enum AggregationMethod  { vlAggregationMax, vlAggregationAverage, vlAggregationLse} ;

  vl::ErrorCode
  nnaggregation_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor data,
                    vl::Tensor pcoords,
                    AggregationMethod method,
                    double r) ;

  vl::ErrorCode
  nnaggregation_backward(vl::Context& context,
                     vl::Tensor derData,
                     vl::Tensor data,
                     vl::Tensor pcoords,
                     vl::Tensor derOutput,
                     AggregationMethod  method,
                     double r) ;
}

#endif /* defined(__vl__nnaggregation__) */
