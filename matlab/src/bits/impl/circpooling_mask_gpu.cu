// @file circ_pooling_gpu.cu
// @brief Polar Pooling block implementation (gpu)
// @author Iván González Díaz



#define EQUAL_SPLIT 1
#include "circpooling_mask.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>

extern __device__ double atomicAdd(double* address, double val);

/* ---------------------------------------------------------------- */
/*                                      circpooling_mask_max_kernel */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
circpooling_mask_max_kernel
(T* pooled,
 const T* data,
 const T* pcoords,
 const int pooledRings,
 const int pooledAngles,
 const int pooledVolume,
 const int height,
 const int width,
 const int depth,
 const int channels,
 const int poolRings,
 const int poolAngles,
 const float overlapRing,
 const float overlapAngle,
 const int padLeft,
 const int padTop)
{


  T radius, angle,minr,maxr,mina[2],maxa[2];

  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
	int pz = pooledIndex / (pooledAngles*pooledRings) ;
	int relLoc = pooledIndex - pz*pooledAngles*pooledRings;
	T ir = (T)(relLoc % pooledRings);
	T ia = (T)(relLoc / pooledRings);
    int pim = pz / channels;
    int offsetAngle = width*height;
    

    data += pz * (width*height) ;
    pcoords += pim * (width*height*2) ;
    //We divide angles non-uniformly, generating rings with approximately equal number of points
    if(EQUAL_SPLIT>0)
    {
    	maxr = sqrt((ir+1)/((T)pooledRings));
    	minr = sqrt(ir/((T)pooledRings));
    }
    //Divide angles uniformly, so outer rings contain more points
    else{
    	//Linear split
    	maxr = (ir+1)/((T)pooledRings);
    	minr = ir/((T)pooledRings);
    }
    maxr=maxr+T(0.0001);
    //Now apply the overlap
    maxr+=(maxr-minr)*overlapRing;
    minr-=(maxr-minr)*overlapRing;
    minr=max(minr,T(0));

    maxa[0] = (ia+1)*2*PI/((T)pooledAngles);
    mina[0] = ia*2*PI/((T)pooledAngles);
    //Now apply the overlap
    maxa[0]+=(maxa[0]-mina[0])*overlapAngle;
    mina[0]-=(maxa[0]-mina[0])*overlapAngle;

    //If we need to add a second angle segment
    if(maxa[0]>(2*PI)){
    	mina[1]=0;
    	maxa[1]=maxa[0]-2*PI;
    	maxa[0]=2*PI;
    }
    else if(mina[0]<0){
    	mina[1]=2*PI+mina[0];
    	maxa[1]=2*PI;
    	mina[0]=0;
    }
    else{
    	mina[1]=1;
    	maxa[1]=-1;
    }
    maxa[0]=maxa[0]+T(0.00001);
    maxa[1]=maxa[1]+T(0.00001);


    //Now check the value we are looking for
    T bestValue = -10000 ;
    int cont=0;
    for (int y = 0; y < height; ++y) {
    	for (int x = 0; x < width; ++x) {
    		radius = pcoords[x * height + y];
    		if(radius>=minr && radius<=maxr){
    			//We get the angle
    			angle = pcoords[offsetAngle + x * height + y];
    			if((angle>=mina[0] && angle <= maxa[0]) || (angle>=mina[1] && angle <= maxa[1])){
    				bestValue = max(bestValue, data[x * height + y]) ;
    				cont++;
    			}
    		}
    	}	
    }
  
  if(cont==0)
    	pooled[pooledIndex] = 0;
    else
    	pooled[pooledIndex] = bestValue;
  }
}

/* ---------------------------------------------------------------- */
/*                                 circpooling_mask_average_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
circpooling_mask_average_kernel
(T* pooled,
 const T* data,
 const T* pcoords,
 const int pooledRings,
 const int pooledAngles,
 const int pooledVolume,
 const int height,
 const int width,
 const int depth,
 const int channels,
 const int poolRings,
 const int poolAngles,
 const float overlapRing,
 const float overlapAngle,
 const int padLeft,
 const int padTop)
{


	T radius, angle,minr,maxr,mina[2],maxa[2];

	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (pooledIndex < pooledVolume) {
		int pz = pooledIndex / (pooledAngles*pooledRings) ;
		int relLoc = pooledIndex - pz*pooledAngles*pooledRings;
		T ir = (T)(relLoc % pooledRings);
		T ia = (T)(relLoc / pooledRings);
		int pim = pz / channels;
		int offsetAngle = width*height;


		data += pz * (width*height) ;
		pcoords += pim * (width*height*2) ;
		if(EQUAL_SPLIT>0)
		{
			maxr = sqrt((ir+1)/((T)pooledRings));
			minr = sqrt(ir/((T)pooledRings));
		}
		else{
			//Linear split
			maxr = (ir+1)/((T)pooledRings);
			minr = ir/((T)pooledRings);
		}
		maxr=maxr+T(0.0001);
		//Now apply the overlap
		maxr+=(maxr-minr)*overlapRing;
		minr-=(maxr-minr)*overlapRing;
		minr=max(minr,T(0));
				
		maxa[0] = (ia+1)*2*PI/((T)pooledAngles);
		mina[0] = ia*2*PI/((T)pooledAngles);
		//Now apply the overlap
		maxa[0]+=(maxa[0]-mina[0])*overlapAngle;
		mina[0]-=(maxa[0]-mina[0])*overlapAngle;
		
		//If we need to add a second angle segment
		if(maxa[0]>(2*PI)){
			mina[1]=0;
			maxa[1]=maxa[0]-2*PI;
			maxa[0]=2*PI;
		}
		else if(mina[0]<0){
			mina[1]=2*PI+mina[0];
			maxa[1]=2*PI;
			mina[0]=0;
		}
		else{
			mina[1]=1;
			maxa[1]=-1;
		}
		maxa[0]=maxa[0]+T(0.00001);
		maxa[1]=maxa[1]+T(0.00001);

		//Now check the value we are looking for
		T accum = 0;
		T cont = 0;//, cont_ant=0;
		//int initx=0;
		for (int y = 0; y < height; ++y) {
			//initx=0;
			for (int x = 0; x < width; ++x) {
				radius = pcoords[x * height + y];
				if(radius>=minr && radius<maxr){
					//We get the angle
					angle = pcoords[offsetAngle + x * height + y];
					if((angle>=mina[0] && angle < maxa[0]) || (angle>=mina[1] && angle < maxa[1])){
						accum += data[x * height + y] ;
						cont++;
					}
				}
				
			}
		}

		cont=cont+0.00000001;
			
		cont=T(width*height)/T(poolRings*poolAngles);
		pooled[pooledIndex] = accum / cont ;
	}
}

/* ---------------------------------------------------------------- */
/*                             circpooling_mask_max_backward_kernel */
/* ---------------------------------------------------------------- */
template<typename T> __global__ void
circpooling_mask_max_backward_kernel
(T* derData,
 const T* data,
 const T* pcoords,
 const T* derPooled,
 const int pooledRings,
 const int pooledAngles,
 const int pooledVolume,
 const int height,
 const int width,
 const int depth,
 const int channels,
 const int poolRings,
 const int poolAngles,
 const float overlapRing,
 const float overlapAngle,
 const int padLeft,
 const int padTop)
{


	T radius, angle,minr,maxr,mina[2],maxa[2];

	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (pooledIndex < pooledVolume) {
		int pz = pooledIndex / (pooledAngles*pooledRings) ;
		int relLoc = pooledIndex - pz*pooledAngles*pooledRings;
		T ir = (T)(relLoc % pooledRings);
		T ia = (T)(relLoc / pooledRings);
		int pim = pz / channels;
		int offsetAngle = width*height;


		data += pz * (width*height) ;
		pcoords += pim * (width*height*2) ;
		derData += pz * (width*height) ;
		if(EQUAL_SPLIT>0)
		{
			maxr = sqrt((ir+1)/((T)pooledRings));
			minr = sqrt(ir/((T)pooledRings));
		}
		else{
			//Reparto lineal
			maxr = (ir+1)/((T)pooledRings);
			minr = ir/((T)pooledRings);
		}
		maxr=maxr+T(0.0001);
		//Now apply the overlap
		maxr+=(maxr-minr)*overlapRing;
		minr-=(maxr-minr)*overlapRing;
		minr=max(minr,T(0));

		maxa[0] = (ia+1)*2*PI/((T)pooledAngles);
		mina[0] = ia*2*PI/((T)pooledAngles);
		//Now apply the overlap
		maxa[0]+=(maxa[0]-mina[0])*overlapAngle;
		mina[0]-=(maxa[0]-mina[0])*overlapAngle;

		//If we need to add a second angle segment
		if(maxa[0]>(2*PI)){
			mina[1]=0;
			maxa[1]=maxa[0]-2*PI;
			maxa[0]=2*PI;
		}
		else if(mina[0]<0){
			mina[1]=2*PI+mina[0];
			maxa[1]=2*PI;
			mina[0]=0;
		}
		else{
			mina[1]=1;
			maxa[1]=-1;
		}
		maxa[0]=maxa[0]+T(0.00001);
		maxa[1]=maxa[1]+T(0.00001);


		//Now check the value we are looking for
		T bestValue = -10000 ;
		int bestIndex = -1;

		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				radius = pcoords[x * height + y];
				if(radius>=minr && radius<=maxr){
					//We get the angle
					angle = pcoords[offsetAngle + x * height + y];
					if((angle>=mina[0] && angle <= maxa[0]) || (angle>=mina[1] && angle <= maxa[1])){
						  int index = x * height + y;
						  if (data[x * height + y] > bestValue) {
							  bestValue = data[index] ;
							  bestIndex = index ;
						  }
					  }
				  }
			  }
		  }	
    
      atomicAdd(derData + bestIndex, derPooled[pooledIndex]) ;
  }
}

/* ---------------------------------------------------------------- */
/*                         circpooling_mask_average_backward_kernel */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
circpooling_mask_average_backward_kernel(T* derData,
		const T* pcoords,
		const T* derPooled,
		const int pooledRings,
		const int pooledAngles,
		const int pooledVolume,
		const int height,
		const int width,
		const int depth,
		const int channels,
		const int poolRings,
		const int poolAngles,
		const float overlapRing,
		const float overlapAngle,
		const int padLeft,
		const int padTop)
{


	T radius, angle,minr,maxr,mina[2],maxa[2];

	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (pooledIndex < pooledVolume) {
		int pz = pooledIndex / (pooledAngles*pooledRings) ;
		int relLoc = pooledIndex - pz*pooledAngles*pooledRings;
		T ir = (T)(relLoc % pooledRings);
		T ia = (T)(relLoc / pooledRings);
		int pim = pz / channels;
		int offsetAngle = width*height;
		
		//we need to advance in derData
		derData += pz * (width*height) ;
		pcoords += pim * (width*height*2) ;
		
		T counter = T(width*height)/T(pooledRings*pooledAngles);
		
		
		if(EQUAL_SPLIT>0)
		{
			maxr = sqrt((ir+1)/((T)pooledRings));
			minr = sqrt(ir/((T)pooledRings));
		}
		else{
			//Reparto lineal
			maxr = (ir+1)/((T)pooledRings);
			minr = ir/((T)pooledRings);
		}
		maxr=maxr+T(0.0001);
		//Now apply the overlap
		maxr+=(maxr-minr)*overlapRing;
		minr-=(maxr-minr)*overlapRing;
		minr=max(minr,T(0));

		maxa[0] = (ia+1)*2*PI/((T)pooledAngles);
		mina[0] = ia*2*PI/((T)pooledAngles);
		//Now apply the overlap
		maxa[0]+=(maxa[0]-mina[0])*overlapAngle;
		mina[0]-=(maxa[0]-mina[0])*overlapAngle;

		//If we need to add a second angle segment
		if(maxa[0]>(2*PI)){
			mina[1]=0;
			maxa[1]=maxa[0]-2*PI;
			maxa[0]=2*PI;
		}
		else if(mina[0]<0){
			mina[1]=2*PI+mina[0];
			maxa[1]=2*PI;
			mina[0]=0;
		}
		else{
			mina[1]=1;
			maxa[1]=-1;
		}
		maxa[0]=maxa[0]+T(0.00001);
		maxa[1]=maxa[1]+T(0.00001);

		counter=max(counter,T(1));
		//T cont=0;
		//cont_ant=0;
		for (int y = 0; y < height; ++y) {
			//initx=0;
			for (int x = 0; x < width; ++x) {
				radius = pcoords[x * height + y];
				if(radius>=minr && radius<maxr){
					//We get the angle
					angle = pcoords[offsetAngle + x * height + y];
					if((angle>=mina[0] && angle < maxa[0]) || (angle>=mina[1] && angle < maxa[1])){
						atomicAdd(&(derData[x * height + y]), derPooled[pooledIndex]/counter) ;
						
					}
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
  struct circpooling_mask_max<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            type const* pcoords,
            size_t height, size_t width, size_t depth, size_t channels,
            size_t poolRings, size_t poolAngles,
            float overlapRing, float overlapAngle,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      int pooledAngles = poolAngles;
      int pooledRings = poolRings ;
      int pooledVolume = pooledRings * pooledAngles * depth ;

      circpooling_mask_max_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (pooled, data, pcoords,
       pooledRings, pooledAngles, pooledVolume,
       height, width, depth, channels,
       poolRings, poolAngles,
	   overlapRing, overlapAngle,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* pcoords,
             type const* derOutput,
             size_t height, size_t width, size_t depth, size_t channels,
             size_t poolRings, size_t poolAngles,
             float overlapRing, float overlapAngle,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight)
    {
    	int pooledAngles = poolAngles;
    	int pooledRings = poolRings ;
    	int pooledVolume = pooledRings * pooledAngles * depth ;

      circpooling_mask_max_backward_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, data, pcoords,derOutput,
       pooledRings, pooledAngles, pooledVolume,
       height, width, depth, channels,
       poolRings, poolAngles,
	   overlapRing, overlapAngle,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // pooling_max

  template <typename type>
  struct circpooling_mask_average<vl::VLDT_GPU, type>
  {

    static vl::ErrorCode
    forward(type* pooled,
    		type const* data,
    		type const* pcoords,
    		size_t height, size_t width, size_t depth, size_t channels,
    		size_t poolRings, size_t poolAngles,
    		float overlapRing, float overlapAngle,
    		size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
    	int pooledAngles = poolAngles;
    	int pooledRings = poolRings ;
    	int pooledVolume = pooledRings * pooledAngles * depth ;

      circpooling_mask_average_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (pooled, data, pcoords,
    	pooledRings, pooledAngles, pooledVolume,
    	height, width, depth, channels,
    	poolRings, poolAngles,
		overlapRing, overlapAngle,
    	padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    static vl::ErrorCode
    backward(type* derData,
    		 type const* pcoords,
             type const* derPooled,
             size_t height, size_t width, size_t depth, size_t channels,
             size_t poolRings, size_t poolAngles,
             float overlapRing, float overlapAngle,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight)
    {

      int pooledAngles = poolAngles;
      int pooledRings = poolRings ;
      int dataVolume = poolAngles * poolRings * depth ;
          	
      circpooling_mask_average_backward_kernel<type>
      <<< divideAndRoundUp(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, pcoords, derPooled,
       pooledRings, pooledAngles, dataVolume,
       height, width, depth, channels,
       poolRings, poolAngles,
	   overlapRing,overlapAngle,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // pooling_average

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::circpooling_mask_max<vl::VLDT_GPU, float> ;
template struct vl::impl::circpooling_mask_average<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::circpooling_mask_max<vl::VLDT_GPU, double> ;
template struct vl::impl::circpooling_mask_average<vl::VLDT_GPU, double> ;
#endif

