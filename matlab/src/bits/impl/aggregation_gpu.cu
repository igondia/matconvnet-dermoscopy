// @file aggregation_gpu.cu
// @brief Aggregation block implementation (gpu)
// @author Iván González Díaz

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "aggregation.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>

extern __device__ double atomicAdd(double* address, double val);

/* ---------------------------------------------------------------- */
/*                                          aggregation_max_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
aggregation_max_kernel
(T* pooled,
 const T* data,
 const T* validos,
 const int pooledVolume,
 const int height,
 const int width,
 const int depth,
 const int channels,
 const T r)
{
  
  
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
	int pz = pooledIndex;
	int pim = pz / channels;
	
    data += pz * (width*height) ;
    validos += pim * (width*height) ;
    
    T bestValue = -100000;
    for (int y = 0; y < height; ++y) {
    	for (int x = 0; x < width; ++x) {
    		if(validos[x * height + y]>0)
    			bestValue = max(bestValue, data[x * height + y]) ;
    	}	
    }
    pooled[pooledIndex] = bestValue ;
  }
}

/* ---------------------------------------------------------------- */
/*                                      aggregation_average_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
aggregation_average_kernel
(T* pooled,
 const T* data,
 const T* validos,
 const int pooledVolume,
 const int height,
 const int width,
 const int depth,
 const int channels,
 const T r)
{
	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (pooledIndex < pooledVolume) {
		
		int pz = pooledIndex;
		int pim = pz / channels;

		data += pz * (width*height) ;
		validos += pim * (width*height) ;

		T accum = 0;
		T counter=0;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(validos[x * height + y]>0){
					accum += data[x * height + y];
					counter=counter+1;
				}
			}	
		}
		counter=max(counter,T(1));
		pooled[pooledIndex] = accum/counter ;
	}
}


/* ---------------------------------------------------------------- */
/*                                          aggregation_lse_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
aggregation_lse_kernel
(T* pooled,
 const T* data,
 const T* validos,
 const int pooledVolume,
 const int height,
 const int width,
 const int depth,
 const int channels,
 const T r)
{
	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (pooledIndex < pooledVolume) {
		
		int pz = pooledIndex;
		int pim = pz / channels;

		data += pz * (width*height) ;
		validos += pim * (width*height) ;

		T accum = 0;
		T counter=0;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(validos[x * height + y]>0){
					accum += exp(r*data[x * height + y]);
					counter=counter+1;
				}
			}	
		}
		counter=max(counter,T(1));
		pooled[pooledIndex] = log(accum/counter)/r;
	}
}

/* ---------------------------------------------------------------- */
/*                                             pooling_max_backward */
/* ---------------------------------------------------------------- */

#ifdef VLNN_CAFFELIKE_BPPOOL
// In order to be able to use this, BP would need to have access to both
// bottom data and pooled data (currently only passed bottom data...)
template <typename T> __global__ void
pooling_max_backward_with_pooled_data
(T* derData,
 const T* data,
 const T* pooled,
 const T* derPooled,
 const int nthreads,
 const int pooledWidth,
 const int pooledHeight,
 const int width,
 const int height,
 const int depth,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY)
{
  int qr, qa;
  T radius, angle, stepRadius, stepAngle;
	  
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    int x = index % width;
    int y = (index / width) % height;
    int z = (index / width / height) % depth;
    int py1 = (y < poolHeight) ? 0 : (y - poolHeight) / strideY + 1;
    int py2 = min(y / strideY + 1, pooledHeight);
    int px1 = (x < poolWidth) ? 0 : (x - poolWidth) / strideX + 1;
    int px2 = min(x / strideX + 1, pooledWidth);
    T gradient = 0;
    T datum = data[(z * height + y) * width + x];
    pooled += z * pooledHeight * pooledWidth;
    dzdy += z * pooledHeight * pooledWidth;
    for (int py = py1; py < py2; ++py) {
      for (int px = px1; px < px2; ++px) {
        gradient += dzdy[py * pooledWidth + px] *
        (datum == pooled[py * pooledWidth + px]);
      }
    }
    dzdx[index] = gradient;
  }
}
#endif

// an implementation of atomicAdd() for double (really slow)
/*__device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}*/

template<typename T> __global__ void
aggregation_max_backward_kernel
(T* derData,
 const T* data,
 const T* validos,
 const T* derPooled,
 const int pooledVolume,
 const int height,
 const int width,
 const int depth,
 const int channels,
 const T r)
{
	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (pooledIndex < pooledVolume) {

		int pz = pooledIndex;
		int pim = pz / channels;

		data += pz * (width*height) ;
		validos += pim * (width*height) ;
		derData += pz * (width*height) ;
		
		T bestValue = -10000 ;
		int bestIndex = -1;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(validos[x * height + y]>0){
					int index = x * height + y;
					if (data[x * height + y] > bestValue) {
						bestValue = data[index] ;
						bestIndex = index ;
					}
				}
			}	
		}
		//We do not need atomicAdd as each update is made on a different memory location 
		derData[bestIndex]=derPooled[pooledIndex];
	}
}


/* ---------------------------------------------------------------- */
/*                                     aggregation_average_backward */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
aggregation_average_backward_kernel(T* derData,
		const T* validos,
		const T* derPooled,
		const int pooledVolume,
		const int height,
		const int width,
		const int depth,
		const int channels,
		const T r)
{
	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (pooledIndex < pooledVolume) {

		int pz = pooledIndex;
		int pim = pz / channels;

		derData += pz * (width*height);
		validos += pim * (width*height) ;
		
		T counter = 0;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(validos[x * height + y]>0){
					counter=counter+1;
				}
			}	
		}
		counter=max(counter,T(1));
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(validos[x * height + y]>0){
					derData[x * height + y]=derPooled[pooledIndex]/counter;
				}
			}	
		}		
	}
}
    

/* ---------------------------------------------------------------- */
/*                                         aggregation_lse_backward */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
aggregation_lse_backward_kernel(T* derData,
		const T* data,
		const T* validos,
		const T* derPooled,
		const int pooledVolume,
		const int height,
		const int width,
		const int depth,
		const int channels,
		const T r)
{
	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (pooledIndex < pooledVolume) {

		int pz = pooledIndex;
		int pim = pz / channels;

		data += pz * (width*height) ;
		derData += pz * (width*height);
		validos += pim * (width*height) ;

		T accumulator= 0;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(validos[x * height + y]>0){
					accumulator=accumulator+exp(r*data[x * height + y]);
				}
			}	
		}
		accumulator=max(accumulator,T(1));
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(validos[x * height + y]>0){
					derData[x * height + y]=derPooled[pooledIndex] * exp(r*data[x * height + y]) / accumulator;
				}
			}	
		}
	}
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct aggregation_max<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            type const* validos,
            size_t height, size_t width, size_t depth, size_t channels,
            type const r)
    {
      int pooledVolume = depth ;

      aggregation_max_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (pooled, data, validos,
       pooledVolume,
       height, width, depth, channels,
       r);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* validos,
             type const* derOutput,
             size_t height, size_t width, size_t depth, size_t channels,
             type const r)
    {
    	int pooledVolume = depth ;

      aggregation_max_backward_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, data, validos,derOutput,
       pooledVolume,
       height, width, depth, channels,
       r);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // pooling_max

  template <typename type>
  struct aggregation_average<vl::VLDT_GPU, type>
  {

    static vl::ErrorCode
    forward(type* pooled,
    		type const* data,
    		type const* validos,
    		size_t height, size_t width, size_t depth, size_t channels,
    		type const r)
    {
    	int pooledVolume = depth ;

      aggregation_average_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (pooled, data, validos,
    	pooledVolume,
    	height, width, depth, channels,
    	r);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    static vl::ErrorCode
    backward(type* derData,
    		 type const* validos,
             type const* derPooled,
             size_t height, size_t width, size_t depth, size_t channels,
             type const r)
    {

      int dataVolume = depth ;
          	
      aggregation_average_backward_kernel<type>
      <<< divideAndRoundUp(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, validos, derPooled,
       dataVolume,
       height, width, depth, channels,
       r);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // pooling_average

  template <typename type>
  struct aggregation_lse<vl::VLDT_GPU, type>
  {
	  static vl::ErrorCode
	  forward(type* pooled,
			  type const* data,
			  type const* validos,
			  size_t height, size_t width, size_t depth, size_t channels,
			  type const r)
	  {
		  int pooledVolume = depth ;

		  aggregation_lse_kernel<type>
		  <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
		  (pooled, data, validos,
				  pooledVolume,
				  height, width, depth, channels,
				  r);

		  cudaError_t status = cudaPeekAtLastError() ;
		  return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
	  }

	  static vl::ErrorCode
	  backward(type* derData,
			  type const* data,
			  type const* validos,
			  type const* derOutput,
			  size_t height, size_t width, size_t depth, size_t channels,
			  type const r)
	  {
		  int pooledVolume = depth ;

		  aggregation_lse_backward_kernel<type>
		  <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
		  (derData, data, validos,derOutput,
				  pooledVolume,
				  height, width, depth, channels,
				  r);

		  cudaError_t status = cudaPeekAtLastError() ;
		  return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
	  }
  } ; // aggregation_lse
  
} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::aggregation_max<vl::VLDT_GPU, float> ;
template struct vl::impl::aggregation_average<vl::VLDT_GPU, float> ;
template struct vl::impl::aggregation_lse<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::aggregation_max<vl::VLDT_GPU, double> ;
template struct vl::impl::aggregation_average<vl::VLDT_GPU, double> ;
template struct vl::impl::aggregation_lse<vl::VLDT_GPU, double> ;
#endif

