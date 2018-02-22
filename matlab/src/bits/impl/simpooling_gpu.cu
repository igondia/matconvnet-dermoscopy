// @file simpooling_gpu.cu
// @brief Simmetry Pooling block implementation (gpu)
// @author Iván González Díaz


/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "simpooling.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>


/* ---------------------------------------------------------------- */
/*                                              sim_pooling_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
simpooling_kernel
(T* pooled,
 const T* data,
 const int poolRings,
 const int poolAngles,
 const int poolVolume,
 const int rings, 
 const int angles, 
 const int depth,
 const int* idx_init)
{
  
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < poolVolume) {
    int pz = pooledIndex / (poolAngles*poolRings) ;
    int relLoc = pooledIndex - pz*poolAngles*poolRings;
    int pa = relLoc / poolRings ;
    T aux;
    
    data += pz * (rings*angles) ;
    
    T scale=T(1.0)/T(rings*poolAngles);
   
    int * idx= new int[2*poolAngles];
    //We shift the matrix to the current pa
    for (int x1 = 0; x1 < angles; ++x1){
    	idx[x1]=idx_init[x1]+pa;
    	if(idx[x1]>=angles)
    		idx[x1]=idx[x1]-angles;
    }
    
    pooled[pooledIndex]=0;
    //Only the half of the angles
    for(int x1 = 0; x1 < poolAngles; ++x1) {
    	//For each ring we compute the differences and accumulate
    	for (int y1 = 0; y1 < rings; ++y1) {
    		//Compute and accumulate the difference among the sectors
    		aux=data[idx[2*x1]*rings + y1]-data[idx[2*x1+1]*rings + y1];
    		pooled[pooledIndex]+=aux*aux*scale;
    	}
    }
    free(idx);
  }
}

/* ---------------------------------------------------------------- */
/*                                              sim_pooling_forward */
/* ---------------------------------------------------------------- */
template<typename T> __global__ void
simpooling_backward_kernel
(T* derData,
 const T* data,
 const T* derPooled,
 const int poolRings,
 const int poolAngles,
 const int poolVolume,
 const int rings, 
 const int angles, 
 const int depth,
 const int* idx_init)
{
    
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  
  
  if (pooledIndex < poolVolume) {
	  
	  int pz = pooledIndex / (poolAngles*poolRings) ;
	  int relLoc = pooledIndex - pz*poolAngles*poolRings;
	  int pa = relLoc / poolRings ;
	  
	  
	  data += pz * rings * angles;
	  derData += pz * rings * angles ;
	  
	  T scale=T(1.0)/T(rings*poolAngles);
	  T sign=1;
	  T aux;
	  //Matrix of indexes for simmetry
	  int * idx= new int[2*poolAngles];
	  //We shift the matrix to the current pa
	  for (int x1 = 0; x1 < angles; ++x1){
		  idx[x1]=idx_init[x1]+pa;
		  if(idx[x1]>=angles)
			  idx[x1]=idx[x1]-angles;
	  }
	  //Only the half of the angles
	  for(int x1 = 0; x1 < poolAngles; ++x1) {
		  //For each ring we compute the differences and accumulate
		  for (int y1 = 0; y1 < rings; ++y1) {
			  //Depending on the sign of the difference => We have to change the values of the mask
			  //In forward we simply do "abs()" => here we need to know the sign to update things
			  aux=data[idx[2*x1]*rings + y1]-data[idx[2*x1+1]*rings + y1];
			  
			  //Update the derivative of the z with respect to the data
			  atomicAdd(derData + idx[2*x1]*rings + y1, aux*derPooled[pooledIndex]*scale) ;
			  atomicAdd(derData + idx[2*x1+1]*rings + y1, -aux*derPooled[pooledIndex]*scale) ;
		  }
	  }
	  free(idx);
  }
}


/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct simpooling<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            size_t rings, size_t angles, size_t depth)
    {
      int poolAngles = angles/2;
      int poolRings = 1 ;
      int poolVolume = poolRings * poolAngles * depth ;
      //Matrix of indexes for simmetry
      int * idx_init,* idx_init_gpu;
      idx_init = (int *)malloc(2*poolAngles*sizeof(int));
      cudaMalloc(&idx_init_gpu, 2*poolAngles*sizeof(int));
      for (int x = 0; x < poolAngles; ++x) {
    	  idx_init[2*x]=x;
    	  idx_init[2*x+1]=angles-1-x;
      }
      cudaMemcpy(idx_init_gpu,idx_init,2*poolAngles*sizeof(int),cudaMemcpyHostToDevice);

      simpooling_kernel<type>
      <<< divideAndRoundUp(poolVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (pooled, data, 
       poolRings, poolAngles, poolVolume,
       rings, angles, depth, idx_init_gpu);

      free(idx_init);
      cudaFree(idx_init_gpu);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* derPooled,
             size_t rings, size_t angles, size_t depth)
    {
    	int poolAngles = angles/2;
    	int poolRings = 1 ;
    	int poolVolume = poolRings * poolAngles * depth ;
    	int * idx_init,* idx_init_gpu;
    	idx_init = (int *)malloc(2*poolAngles*sizeof(int));
    	cudaMalloc(&idx_init_gpu, 2*poolAngles*sizeof(int));
    	for (int x = 0; x < poolAngles; ++x) {
    		idx_init[2*x]=x;
    		idx_init[2*x+1]=angles-1-x;
    	}
    	cudaMemcpy(idx_init_gpu,idx_init,2*poolAngles*sizeof(int),cudaMemcpyHostToDevice);

    	
    	simpooling_backward_kernel<type>
      <<< divideAndRoundUp(poolVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, data, derPooled,
    	poolRings, poolAngles, poolVolume,
    	rings, angles, depth,idx_init_gpu);

    	free(idx_init);
    	cudaFree(idx_init_gpu);

    	
      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // simpooling

  

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::simpooling<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::simpooling<vl::VLDT_GPU, double> ;
#endif

