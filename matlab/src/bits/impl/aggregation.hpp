// @file aggregation.hpp
// @brief Aggregation block headers
// @author Iván González Díaz


/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_NNAGGREGATION_H
#define VL_NNAGGREGATION_H
#define PI 3.14159265
#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::DeviceType dev, typename type>
  struct aggregation_max {
    typedef type data_type ;

    static vl::ErrorCode
    forward(data_type* output,
            data_type const* data,
            data_type const* pcoords,
            size_t height, size_t width, size_t depth,size_t channels,
            data_type r) ;

    static vl::ErrorCode
    backward(data_type* derData,
             data_type const* data,
             data_type const* pcoords,
             data_type const* derOutput,
             size_t height, size_t width, size_t depth,size_t channels,
             data_type r) ;
  } ;

  template<vl::DeviceType dev, typename type>
  struct aggregation_average {
    typedef type data_type ;

    static vl::ErrorCode
    forward(data_type* output,
            data_type const* data,
            data_type const* pcoords,
            size_t height, size_t width, size_t depth,size_t channels,
            data_type r) ;

    static vl::ErrorCode
    backward(type* derData,
    		 type const* pcoords,
             type const* derOutput,
             size_t height, size_t width, size_t depth,size_t channels,
             data_type r) ;
  } ;

  template<vl::DeviceType dev, typename type>
    struct aggregation_lse {
      typedef type data_type ;

      static vl::ErrorCode
      forward(data_type* output,
              data_type const* data,
              data_type const* pcoords,
              size_t height, size_t width, size_t depth,size_t channels,
              data_type r) ;

      static vl::ErrorCode
      backward(type* derData,
    		   type const* data,
    		   type const* pcoords,
               type const* derOutput,
               size_t height, size_t width, size_t depth,size_t channels,
               data_type r) ;
    } ;

} }

#endif /* defined(VL_POOLING_H) */
