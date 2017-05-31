// @file weakloss.hpp
// @brief Weak Loss block headers
// @author Iván González Díaz

#ifndef VL_WEAKLOSS_H
#define VL_WEAKLOSS_H
#define PI 3.14159265
#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::DeviceType dev, typename type>
  struct weakloss {
    typedef type data_type ;

    static vl::ErrorCode
    forward(data_type* output,
            data_type const* data,
            data_type const* pcoords,
            data_type const* labels,
            data_type * lambda,
            data_type const* A,
            data_type const* b,
            data_type const* beta,
            float const maxLambda,
            size_t height, size_t width, size_t channels,size_t numIm) ;

    static vl::ErrorCode
    backward(data_type* derData,
             data_type const* data,
             data_type const* pcoords,
             data_type const* labels,
             data_type * lambda,
             data_type const* derOutput,
             data_type const* A,
             data_type const* b,
             data_type const* beta,
			 float const maxLambda,
             size_t height, size_t width, size_t channels, size_t numIm) ;

  } ;



} }

#endif /*VL_WEAKLOSS_H*/
