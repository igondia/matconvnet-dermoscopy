// @file circ_pooling.hpp
// @brief Polar Pooling block headers
// @author Iván González Díaz


#ifndef VL_NNCIRCPOOLING_MASK_H
#define VL_NNCIRCPOOLING_MASK_H
#define PI 3.141593
#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::DeviceType dev, typename type>
  struct circpooling_mask_max {
    typedef type data_type ;

    static vl::ErrorCode
    forward(data_type* output,
            data_type const* data,
	    type const* pcoords,
            size_t height, size_t width, size_t depth,size_t channels,
            size_t poolRings, size_t poolAngles,
            float overlapRing, float overlapAngle,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

    static vl::ErrorCode
    backward(data_type* derData,
             data_type const* data,
	     type const* pcoords,
             data_type const* derOutput,
             size_t height, size_t width, size_t depth,size_t channels,
             size_t poolRings, size_t poolAngles,
			 float overlapRing, float overlapAngle,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
  } ;

  template<vl::DeviceType dev, typename type>
  struct circpooling_mask_average {
    typedef type data_type ;

    static vl::ErrorCode
    forward(data_type* output,
            data_type const* data,
            type const* pcoords,
            size_t height, size_t width, size_t depth,size_t channels,
            size_t poolRings, size_t poolAngles,
			float overlapRing, float overlapAngle,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

    static vl::ErrorCode
    backward(type* derData,
    		 type const* pcoords,
             type const* derOutput,
             size_t height, size_t width, size_t depth,size_t channels,
             size_t poolRings, size_t poolAngles,
			 float overlapRing, float overlapAngle,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
  } ;

} }

#endif /* defined(VL_POOLING_H) */
