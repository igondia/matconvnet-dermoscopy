// @file aggregation_cpu.cpp
// @brief Aggregation block implementation (cpu)
// @author Iván González Díaz

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "aggregation.hpp"
#include "../data.hpp"
#include <algorithm>
#include <limits>
#include <math.h>
#include "../mexutils.h"

/* ---------------------------------------------------------------- */
/*                                               Max pooling helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_max
{

  inline acc_max(type r, type derOutput = 0)
  :
  value(-std::numeric_limits<type>::infinity()),
  derOutput(derOutput),
  derDataActivePt(NULL)
  { }

  inline void init(type r, type aderOutput = 0){
    value=-std::numeric_limits<type>::infinity();
    derOutput = aderOutput;
    derDataActivePt = NULL;
  }


  inline void accumulate_forward(type x) {
    value = std::max(value, x) ;
  }

  inline void accumulate_scale(type x){
     }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    type x = *data ;
    if (x > value) {
      value = x ;
      derDataActivePt = derDataPt ;
    }
  }

  inline type done_forward() const {
    return value ;
  }

  inline void done_backward() const {
    if (derDataActivePt) { *derDataActivePt += derOutput ; }
  }

  type value ;
  type derOutput ;
  type* derDataActivePt ;
} ;

/* ---------------------------------------------------------------- */
/*                                           Average pooling helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_sum
{
  inline acc_sum(type r, type derOutput = 0)
  :
  value(0),
  scale(0),//type(1)/type(poolRings*poolAngles)),
  derOutput(derOutput)
  { }

  inline void init(type r, type aderOutput = 0){
	  scale= 0;//type(1.0/(poolRings*poolAngles));
	  value = 0;
	  derOutput = aderOutput;
  }

  inline void accumulate_forward(type x) {
    value += x ;
    scale += type(1.0) ;
  }

  inline void accumulate_scale(type x){
  	  scale += type(1.0) ;
    }

  /* note: data is unused */
  inline void accumulate_backward(type const* data, type* derDataPt) {
	 //mexPrintf("scale %f\n",scale);mexEvalString("drawnow");
	*derDataPt += (derOutput / (scale + type(0.00001)));
  }

  inline type done_forward() const {
	  /*if(scale==type(0.0)){
		  mexPrintf("Warning!!!: Circpooling without data in a cell!!!\n");mexEvalString("drawnow");
		  return type(-2000);
	  }*/
	  return value / (scale + type(0.00001));
  }

  inline void done_backward() const { }

  type value ;
  type derOutput ;
  type scale ;
} ;


template <typename type>
struct acc_lse
{
  inline acc_lse(type r, type derOutput = 0)
  :
  value(0),
  scale(0),//type(1)/type(poolRings*poolAngles)),
  r_param(r),
  derOutput(derOutput)
  { }

  inline void init(type r, type aderOutput = 0){
	  scale= 0;//type(1.0/(poolRings*poolAngles));
	  value = 0;
	  r_param = r;
	  derOutput = aderOutput;
  }

  inline void accumulate_forward(type x) {
    value += exp(r_param*x) ;
    scale += type(1.0) ;
  }

  inline void accumulate_scale(type x){
  	  scale += exp(r_param*x) ;
    }

  /* note: data is unused */
  inline void accumulate_backward(type const* data, type* derDataPt) {
	type x = *data;
    *derDataPt += (derOutput * exp(r_param*x) / scale);
  }

  inline type done_forward() const {
	  type output = log(value/scale)/r_param;
	  return output;
  }

  inline void done_backward() const { }

  type value ;
  type r_param;
  type derOutput ;
  type scale ;
} ;

/* ---------------------------------------------------------------- */
/*                                                pooling_*_forward */
/* ---------------------------------------------------------------- */

/*
 Reverse accumulation style (better for writing).
 - pick an input coordiante xi; goal is to compute dz/dxi
 - look for all the pools Pj that cointain xi
 -  compute dfj/dxi (Pj)
 -  accumulate into dz/dxi += dz/dfj dfj/dxi (Pj)

 The advantage of this method is that dz/dxi can be processed in parallel
 without conflicts from other threads writing on different dz/dxi. The
 disadvantage is that for eac xi we need to know dfj/dxi (Pj) for all
 the pools Pj that contain xi. Barring special cases (e.g. linear) this
 usually requires additional information to be available. For instance,
 for max pooling knowing the output in addition to the input of the
 pooling operator.

 Direct accumulation style.
 - pick an output coordiante fj and its pool Pj
 - for all the input point xi in the pool Pj
 - compute dfj/dxi (Pj)
 - accumulate to dz/dxi += dz/dfj dfj/dxi (Pj)

 The difference with the last method is that different output pools Pj
 will share several input pixels xi; hence this will cause write conflicts if
 Pj are processed in parallel.
 */

template<typename type, typename Accumulator> static inline void
aggregation_forward_cpu(type* pooled,
                    type const* data,
                    type const* validos,
                    size_t height, size_t width, size_t depth, size_t channels,
                    type r)
{
	int contChannels=0;

	for (int z = 0; z < depth; ++z) {
		contChannels++;

		Accumulator acc(r) ;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(validos[x * height + y]>0)
					acc.accumulate_forward(data[x * height + y]) ;
			}
		}
		pooled[z] = acc.done_forward() ;

		data += width*height ;
		if(contChannels==channels){
			validos += width*height;
			contChannels=0;
		}

	}

}

/* ---------------------------------------------------------------- */
/*                                               pooling_*_backward */
/* ---------------------------------------------------------------- */

/*
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */

/* Todo: transpose */

template<typename type, typename Accumulator> static inline void
aggregation_backward_cpu(type* derData,
                     type const* data,
                     type const* validos,
                     type const* derPooled,
                     size_t height, size_t width, size_t depth, size_t channels,
                     type r)
{
	int contChannels=0;
	for (int z = 0; z < depth; ++z) {
		contChannels++;
		Accumulator acc(r,derPooled[z]) ;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(validos[x * height + y]>0)
					acc.accumulate_scale(data[x * height + y]);
			}
		}
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(validos[x * height + y]>0)
					acc.accumulate_backward(&data[x * height + y],&derData[x * height + y]) ;
			}
		}

		acc.done_backward() ;

		data += width*height ;
		derData += width*height ;
		if(contChannels==channels){
			validos += width*height ;
		}
	}
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct aggregation_max<vl::VLDT_CPU, type>
  {
    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            type const* validos,
            size_t height, size_t width, size_t depth, size_t channels,
            type r)
    {
      aggregation_forward_cpu<type, acc_max<type> > (pooled,
                                                 data,
                                                 validos,
                                                 height, width, depth, channels,
                                                 r) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* validos,
             type const* derOutput,
             size_t height, size_t width, size_t depth, size_t channels,
             type r)
    {
      aggregation_backward_cpu<type, acc_max<type> > (derData,
                                                  data, validos,derOutput,
                                                  height, width, depth, channels,
                                                  r) ;
      return VLE_Success ;
    }
  } ; // aggregation_max

  template <typename type>
  struct aggregation_average<vl::VLDT_CPU, type>
  {

    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            type const* validos,
            size_t height, size_t width, size_t depth, size_t channels,
            type r)
    {
      aggregation_forward_cpu<type, acc_sum<type> > (pooled,
                                                 data,
                                                 validos,
                                                 height, width, depth, channels,
                                                 r) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
    		 type const* validos,
             type const* derPooled,
             size_t height, size_t width, size_t depth, size_t channels,
             type r)
    {
      aggregation_backward_cpu<type, acc_sum<type> > (derData,
    		  	  	  	  	  	  	  	  	  	  NULL, validos, derPooled,
                                                  height, width, depth, channels,
                                                  r) ;
      return VLE_Success ;
    }
  } ; // aggregation_average


  template <typename type>
    struct aggregation_lse<vl::VLDT_CPU, type>
    {

      static vl::ErrorCode
      forward(type* pooled,
              type const* data,
              type const* validos,
              size_t height, size_t width, size_t depth, size_t channels,
              type r)
      {
        aggregation_forward_cpu<type, acc_lse<type> > (pooled,
                                                   data,
                                                   validos,
                                                   height, width, depth, channels,
                                                   r) ;
        return VLE_Success ;
      }

      static vl::ErrorCode
      backward(type* derData,
    		 type const* data,
      		 type const* validos,
             type const* derPooled,
             size_t height, size_t width, size_t depth, size_t channels,
             type r)
      {
        aggregation_backward_cpu<type, acc_lse<type> > (derData,
        											data, validos, derPooled,
                                                    height, width, depth, channels,
                                                    r) ;
        return VLE_Success ;
      }
    } ; // aggregation_lse

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::aggregation_max<vl::VLDT_CPU, float> ;
template struct vl::impl::aggregation_average<vl::VLDT_CPU, float> ;
template struct vl::impl::aggregation_lse<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::aggregation_max<vl::VLDT_CPU, double> ;
template struct vl::impl::aggregation_average<vl::VLDT_CPU, double> ;
template struct vl::impl::aggregation_lse<vl::VLDT_CPU, double> ;
#endif

