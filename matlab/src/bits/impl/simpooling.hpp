// @file simpooling.hpp
// @brief Simmetry Pooling block headers
// @author Iván González Díaz

#ifndef VL_SIMPOOLING_H
#define VL_SIMPOOLING_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::DeviceType dev, typename type>
  struct simpooling {
    typedef type data_type ;

    static vl::ErrorCode
    forward(data_type* output,
            data_type const* data,
            size_t rings, size_t angles, size_t depth);


    static vl::ErrorCode
    backward(data_type* derData,
             data_type const* data,
             data_type const* derOutput,
             size_t rings, size_t angles, size_t depth);

  } ;


} }

#endif /* defined(VL_SIMPOOLING_H) */
