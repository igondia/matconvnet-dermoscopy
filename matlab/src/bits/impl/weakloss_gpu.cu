// @file weakloss_gpu.cu
// @brief Weak Loss block implementation (gpu)
// @author Iván González Díaz



#include "weakloss.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <sm_35_atomic_functions.h>
#include "../mexutils.h"

extern __device__ double atomicAdd(double* address, double val);


/* ---------------------------------------------------------------- */
/*                                          weakloss			    */
/* ---------------------------------------------------------------- */

/*Kernel that limits the max value of Lambda*/
template<typename T> __global__ void
limLambda_kernel
(T* lambda,
 const T*labels,
 const T*A,
 float maxLambda,
 const int channels,
 const int numIm,
 const int pooledVolume)
{
	
	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (pooledIndex < pooledVolume) {
		lambda[pooledIndex]=max(lambda[pooledIndex],T(0));
		lambda[pooledIndex]=min(lambda[pooledIndex],T(maxLambda));
	}
}


/*Kernel that counts the total number of valid pixels*/
template<typename T> __global__ void
contValid_kernel
(const T* pcoords,
 int* validPixels,
 const int pooledVolume,
 const int height,
 const int width,
 const int channels,
 const int numIm)
{

	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
	pcoords+=2*pooledIndex*width*height;

	//Set the arrays
	if (pooledIndex < pooledVolume) {
		validPixels[pooledIndex]=0;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				//If they are valid
				if(pcoords[x * height + y]>=0)
					validPixels[pooledIndex]++;
			}
		}
	}
}

/*Weak loss forward kernel*/
template<typename T> __global__ void
weakloss_kernel
(T* LCost,
 T* DCost,
 const T* data,
 const T* pcoords,
 const T* labels,
 const T* lambda,
 T* nlambda,
 const T* A,
 const T* b,
 const T*beta,
 int* validPixels,
 T mu,
 int iter,
 const bool *done,
 const int pooledVolume,
 const int height,
 const int width,
 const int channels,
 const int numIm)
{

	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if (pooledIndex < pooledVolume) {


		//Get the locations of the kernel
		//Image
		int pim=pooledIndex/(height*width);
		//Pixel
		int px=pooledIndex-pim*height*width;
		T validPx=(T)validPixels[pim];

		//Set the arrays on their initial locations
		data += pim*channels*width*height + px ;
		pcoords += pim*2*width*height + px ;
		labels += pim*channels;
		lambda += pim*channels*2;
		nlambda += pim*channels*2;
		LCost +=pim;
		DCost +=pim;
		done +=pim;
		//If the location is valid
		if(*pcoords>=0 && *done==0){
			const T* tdata;
			T *P=(T*)malloc(channels*sizeof(T));
			int l;

			//Get maximum values to limit the inputs
			T maxLambda=0;
			T maxValue=0;
			T weight=0;
			tdata=data;
			for (int z = 0; z < channels; ++z) {
				l=(int)labels[z];
				maxLambda=max(*tdata+A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1],maxLambda);
				maxValue=max(maxValue,*tdata);
				//maxLambda=maxValue;
				tdata+=width*height;
				if(l>0)
					weight+=1-beta[z];
				else
					weight+=beta[z];
			}
			weight=weight*2/channels;
			weight=weight/T(width*height);
			
			//Channels loop
			T sumData=0;
			T normalizer=0;
			T ndata;
			tdata=data;
			for (int z = 0; z < channels; z++) {
				l=(int)labels[z];

				//Non-spatial case: count over all pixels
				if(l!=3){
					P[z]=exp(*tdata+A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1]-maxLambda);
				}
				//Spatial case, we just consider as possible pixels in the lesion boundary
				else{
					//We consider as valid pixels on the boundaries
					if(*pcoords>=0.75)
						P[z]=exp(*tdata+A[2*l]*lambda[2*z]-maxLambda);
					//In the lesion center
					else
						P[z]=exp(*tdata+A[2*l+1]*lambda[2*z+1]-maxLambda);
				}

				normalizer+=P[z];
				sumData+=exp(*tdata-maxValue);
				//Advance one channel
				tdata+=width*height;
			}

			//Update Dual Cost
			if(normalizer>0)
				atomicAdd(DCost,-weight*(log(normalizer)+maxLambda));
			else
				atomicAdd(DCost,-weight*(log(T(0.000001))+maxLambda));

			//Normalization and Lagrangian update
			tdata=data;
			T inc_LCost=0;
			T inc_DCost=0;
			for (int z = 0; z < channels; ++z){
				l=(int)labels[z];
				if(normalizer>0)
					P[z]=P[z]/normalizer;
				else
					P[z]=0;
				if(sumData>0)
					ndata=exp(*tdata-maxValue)/sumData;
				else
					ndata=0;

				//Update Lambda
				if(l!=3){
					T inc=b[2*l]-A[2*l]*P[z];
					atomicAdd(nlambda+2*z,mu*inc/validPx);
					inc=b[2*l+1]-A[2*l+1]*P[z];
					atomicAdd(nlambda+2*z+1,mu*inc/validPx);
					inc_DCost+=lambda[2*z]*b[2*l]+lambda[2*z+1]*b[2*l+1];
				}
				//Spatially constrained case
				else{
					if(*pcoords>=0.75){
						T inc=b[2*l]-A[2*l]*P[z];
						inc_DCost+=lambda[2*z]*b[2*l];
						atomicAdd(&nlambda[2*z],mu*inc/validPx);
					}
					else{
						T inc=b[2*l+1]-A[2*l+1]*P[z];
						inc_DCost+=lambda[2*z+1]*b[2*l+1];
						atomicAdd(&nlambda[2*z+1],mu*inc/validPx);
					}
				}

				//Update Lagrangian Cost
				ndata=max(ndata,T(0.000001));
				if(P[z]>0)
					inc_LCost+=weight*P[z]*log(P[z]/ndata);

				//Advance one channel
				tdata+=width*height;
			}
			atomicAdd(DCost,inc_DCost);
			atomicAdd(LCost,inc_LCost);
			free(P);
		}
	}
}

/*Weak loss backward kernel*/
template <typename T> __global__ void
weakloss_backward_kernel(T* derData,
		const T* data,
		const T* pcoords,
		const T* labels,
		T* lambda,
		const T* derOutput,
		const T* A,
		const T* b,
		const T* beta,
		const float limLambda,
		const int pooledVolume,
		const int height,
		const int width,
		const int channels,
		const int numIm)
{


	int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if (pooledIndex < pooledVolume) {

		//Get the locations of the kernel
		//Image
		int pim=pooledIndex/(height*width);
		//Pixel
		int px=pooledIndex-pim*height*width;
		

		//Set the arrays on their initial locations
		data += pim*channels*width*height + px ;
		pcoords += pim*2*width*height + px ;
		labels += pim*channels;
		lambda += pim*channels*2;
		derData += pim*channels*width*height + px ;

		//If the location is valid
		if(*pcoords>=0){
			int l;

			const T* tdata;
			T *P=(T*)malloc(channels*sizeof(T));

			//Get maximum values to limit the inputs
			T maxLambda=0;
			T maxValue=0;
			T weight=0;
			tdata=data;
			for (int z = 0; z < channels; ++z) {
				l=(int)labels[z];
				maxLambda=max(*tdata+A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1],maxLambda);
				maxValue=max(maxValue,*tdata);
				tdata+=width*height;
				if(l>0)
					weight+=1-beta[z];
				else
					weight+=beta[z];
			}
			weight=2*weight/channels;
			weight=weight/T(width*height);

			//Channels loop
			T sumData=0;
			T normalizer=0;
			T ndata;
			tdata=data;
			for (int z = 0; z < channels; ++z) {
				l=(int)labels[z];
				//Regular case: count over all pixels
				if(l!=3){
					P[z]=exp(*tdata +A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1]-maxLambda);
				}
				//Spatially constrained case: consideronly pixels in the boundary
				else{
					if(*pcoords>=0.75)
						P[z]=exp(*tdata+A[2*l]*lambda[2*z]-maxLambda);
					else
						P[z]=exp(*tdata+A[2*l+1]*lambda[2*z+1]-maxLambda);
				}
				normalizer+=P[z];
				sumData+=exp(*tdata-maxValue);
				//Advance one channel
				tdata+=width*height;
			}

			
			tdata=data;
			for (int z = 0; z < channels; ++z){
				if(normalizer>0)
					P[z]=P[z]/normalizer;
				else{
					P[z]=T(0);
				}
				if(sumData>0)
					ndata=exp(*tdata-maxValue)/sumData;
				else{
					ndata=T(0);
				}

				*derData=weight*derOutput[0]*(ndata-P[z]);
				derData+=width*height;
				tdata+=width*height;
			}
			free(P);
		}
	}
}
/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct weakloss<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    forward(type * LCost,
            type const* data,
            type const* pcoords,
            type const* labels,
            type * lambda,
            type const* A,
            type const* b,
            type const* beta,
            float maxLambda,
            size_t height, size_t width, size_t channels, size_t numIm)

    {

      int iter;
      int K=2*channels,Niter=300,*validPixels;
      type *DCost,*DCost_GPU,*DCost_ant,*LCost_GPU,*BCost;
      type *nlambda,*best_lambda;//,*cpu_labels;
      type mu=10.0,TDCost,TDCost_ant;
      int contFinal=0;
      bool *done;
      int pooledVolume;

      cudaMalloc(&nlambda, K*numIm*sizeof(type));
      cudaMalloc(&validPixels, numIm*sizeof(int));
      cudaMalloc(&DCost_GPU, numIm*sizeof(type));
      cudaMalloc(&LCost_GPU, numIm*sizeof(type));
      cudaMalloc(&done, numIm*sizeof(bool));
      cudaMemset(done,false,numIm*sizeof(bool));

      DCost=(type*)malloc(numIm*sizeof(type));
      DCost_ant=(type*)malloc(numIm*sizeof(type));
      BCost=(type*)malloc(numIm*sizeof(type));
      best_lambda=(type*)malloc(K*numIm*sizeof(type));

      //Count number of valid data
      pooledVolume=(int)numIm;
          contValid_kernel<type>
         	  <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
         	  (pcoords,validPixels,numIm, height, width, channels, numIm);
      
      //Limit lambda values    
      pooledVolume=numIm*K;
      limLambda_kernel<type>
               	  <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
           		  (lambda,labels,A,maxLambda,channels,numIm,numIm*K);

      cudaMemcpy(best_lambda, lambda, K*numIm*sizeof(type),cudaMemcpyDeviceToHost);

      
      //The loop of iterations has to be here as it is operated serially
      for(iter=0;iter<Niter;iter++){
    	  
    	  //cudaMemcpy(DCost_GPU, DCost, numIm*sizeof(type), cudaMemcpyHostToDevice);
    	  cudaMemcpy(nlambda, lambda, K*numIm*sizeof(type),cudaMemcpyDeviceToDevice);
    	  //we reset the partial costs
    	  cudaMemset(LCost_GPU,0,numIm*sizeof(type));
    	  cudaMemset(DCost_GPU,0,numIm*sizeof(type));

    	  pooledVolume = numIm*height*width;
    	  //Forward step
    	  weakloss_kernel<type>
    	  <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
		  (LCost_GPU, DCost_GPU, data, pcoords,
				  labels,lambda,nlambda, A,b,beta,validPixels,mu,iter,done,
				  pooledVolume,
				  height, width, channels, numIm);
    	  pooledVolume=numIm*K;
    	  //Limit the values of lambda
    	  limLambda_kernel<type>
    	  <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
		  (nlambda,labels,A,maxLambda,channels,numIm,numIm*K);

    	  cudaMemcpy(DCost, DCost_GPU, numIm*sizeof(type), cudaMemcpyDeviceToHost);
    	  contFinal=0;
    	  TDCost=0;

    	  //Documents loop to update lambdas
    	  for (int d=0;d<numIm;d++){
    		  type mejora;
    		  bool done_CPU;
    		  cudaMemcpy(&done_CPU,&(done[d]), 1*sizeof(bool), cudaMemcpyDeviceToHost);

    		  if (done_CPU==true){
    			  mejora=0;
    			  TDCost+=DCost_ant[d];
    		  }
    		  else if(iter==0){
    			  mejora=100.0;
    			  TDCost+=DCost[d];
    		  }
    		  else{
    			  mejora=100.0*(DCost[d]-DCost_ant[d])/abs(DCost_ant[d]);
    			  TDCost+=DCost[d];
    		  }

    		  //If we don't improve enough
    		  if(mejora<-0.1 && iter>0 && mu>1e-3){
    		  	 mu=mu*0.5;
    		  	 done_CPU=false;
    		  	 //We set the previous best lambda
    		  	 cudaMemcpy(lambda+K*d,best_lambda+K*d, K*sizeof(type),cudaMemcpyHostToDevice);
    		  }
    		  else if(mejora<=0.1 ){
    			 contFinal++;
    			 done_CPU=true;
    		  }
    		  //if we are improving
    		  else{
    			  //Copy the best option (only in the respective locations)
    			  cudaMemcpy(best_lambda+K*d, lambda+K*d, K*sizeof(type),cudaMemcpyDeviceToHost);
    			  //Copy nlambda to lambda
    			  cudaMemcpy(lambda+K*d, nlambda+K*d, K*sizeof(type),cudaMemcpyDeviceToDevice);
    			  DCost_ant[d]=DCost[d];
    			  cudaMemcpy(&(BCost[d]), &(LCost_GPU[d]), 1*sizeof(type), cudaMemcpyDeviceToHost);
    			  done_CPU=false;
    		  }
    		  cudaMemcpy(&(done[d]), &done_CPU, 1*sizeof(bool), cudaMemcpyHostToDevice);

    	  }

    	  type mejora=100.0*(TDCost-TDCost_ant)/abs(TDCost_ant);
    	  
    	  //After 3 iters without improving
    	  if(contFinal==numIm && iter>4){
    		  break;
    	  }
    	  TDCost_ant=TDCost;

      }
      
      type TBCost=0;
      for (int d=0;d<numIm;d++){
    	  TBCost+=BCost[d];
      }
      cudaMemcpy(LCost, &TBCost, 1*sizeof(type), cudaMemcpyHostToDevice);
      cudaMemcpy(lambda, best_lambda, K*numIm*sizeof(type),cudaMemcpyHostToDevice);
      cudaFree(nlambda);
      cudaFree(validPixels);
      cudaFree(DCost_GPU);
      free(best_lambda);
      free(DCost);
      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;

    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* pcoords,
             type const* labels,
             type * lambda,
             type const* derOutput,
             type const* A,
             type const* b,
             type const* beta,
             float maxLambda,
             size_t height, size_t width, size_t channels, size_t numIm)
    {

    	int pooledVolume = numIm*height*width;
    	int *validPixels;

    	cudaMalloc(&validPixels, numIm*sizeof(int));


    	//Backward kernel
    	weakloss_backward_kernel<type>
    	<<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
		(derData, data, pcoords, labels, lambda, derOutput, A, b, beta,maxLambda,
				pooledVolume,
				height, width, channels, numIm);

    	cudaFree(validPixels);

    	cudaError_t status = cudaPeekAtLastError() ;
    	return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // weakloss

  
} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::weakloss<vl::VLDT_GPU, float> ;


#ifdef ENABLE_DOUBLE
template struct vl::impl::weakloss<vl::VLDT_GPU, double> ;
#endif

