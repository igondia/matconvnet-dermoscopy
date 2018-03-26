// @file weakloss_cpu.cpp
// @brief Weak Loss block implementation (cpu)
// @author Iván González Díaz


#include "weakloss.hpp"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include "../mexutils.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

#define RBorders 0.5

/* ---------------------------------------------------------------- */
/*                                          weakloss			    */
/* ---------------------------------------------------------------- */

/*Kernel that limits the max value of Lambda*/
template<typename type> static inline void
limLambda_kernel_cpu
(type* lambda,
 const type*labels,
 const type*A,
 float maxLambda,
 const int channels,
 const int numIm,
 const int pooledIndex)
{

	lambda[pooledIndex]=std::max(lambda[pooledIndex],type(0));
	lambda[pooledIndex]=std::min(lambda[pooledIndex],type(maxLambda));
}


/*Kernel that counts the total number of valid pixels*/
template<typename type> static void
contValid_kernel_cpu
(const type* pcoords,
 int* validPixels,
 int* validOuterPixels,
 const int pooledIndex,
 const int height,
 const int width,
 const int channels,
 const int numIm)
{

	pcoords+=2*pooledIndex*width*height;

	//Set the arrays
	validPixels[pooledIndex]=0;
	validOuterPixels[pooledIndex]=0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			//If they are valid
			if(pcoords[x * height + y]>=0)
				validPixels[pooledIndex]++;
			if(pcoords[x * height + y]>=RBorders)
				validOuterPixels[pooledIndex]++;
		}
	}

}

/*Weak loss forward kernel*/
template<typename type> static void
weakloss_kernel_cpu
(type* LCost,
 type* DCost,
 const type* data,
 const type* pcoords,
 const type* labels,
 const type* lambda,
 type* nlambda,
 const type* A,
 const type* b,
 const type*beta,
 int* validPixels,
 int* validOuterPixels,
 type mu,
 int iter,
 const bool *done,
 const int pooledIndex,
 const int height,
 const int width,
 const int channels,
 const int numIm)
{

	




		//Get the locations of the kernel
		//Image
		int pim=pooledIndex/(height*width);
		//Pixel
		int px=pooledIndex-pim*height*width;
		type validPx=(type)validPixels[pim];
		type validOuterPx=(type)validOuterPixels[pim];
		type validInnerPx=validPx-validOuterPx;
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
			const type* tdata;
			type *P=(type*)malloc(channels*sizeof(type));
			int l;

			//Get maximum values to limit the inputs
			type maxLambda=0;
			type maxValue=0;
			type weight=0;
			tdata=data;
			for (int z = 0; z < channels; ++z) {
				l=(int)labels[z];
				maxLambda=std::max(*tdata+A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1],maxLambda);
				maxValue=std::max(maxValue,*tdata);
				//maxLambda=maxValue;
				tdata+=width*height;
				if(l>0)
					weight+=1-beta[z];
				else
					weight+=beta[z];
			}
			
			weight=weight*2/channels;

			//Channels loop
			type sumData=0;
			type normalizer=0;
			type ndata;
			tdata=data;
			for (int z = 0; z < channels; z++) {
				l=(int)labels[z];
								
				//Non-spatial case: count over all pixels
				if(l<3){
					P[z]=exp(*tdata+A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1]-maxLambda);
				}
				//Spatial case, we just consider as possible pixels in the lesion boundary
				else{
					//We consider as valid pixels on the boundaries
					if(*pcoords>=RBorders)
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
			if(normalizer>0){
				*DCost-=-weight*(log(normalizer)+maxLambda);
			}
			else{
				*DCost-=weight*(log(type(0.000001)));
			}
			//Normalization and Lagrangian update
			tdata=data;
			type inc_LCost=0;
			type inc_DCost=0;
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
				if(l<3){
					type inc=b[2*l]-A[2*l]*P[z];
					nlambda[2*z]+=mu*inc/validPx;
					inc=b[2*l+1]-A[2*l+1]*P[z];
					nlambda[2*z+1]+=mu*inc/validPx;
					inc_DCost+=lambda[2*z]*b[2*l]+lambda[2*z+1]*b[2*l+1];
				}
				//Spatially constrained case
				else{
					if(*pcoords>=RBorders){
						type inc=b[2*l]-A[2*l]*P[z];
						inc_DCost+=lambda[2*z]*b[2*l];
						nlambda[2*z]+=mu*inc/validOuterPx;
					}
					else{
						type inc=b[2*l+1]-A[2*l+1]*P[z];
						inc_DCost+=lambda[2*z+1]*b[2*l+1];
						nlambda[2*z+1]+=mu*inc/validInnerPx;
					}
				}

				//Update Lagrangian Cost
				ndata=std::max(ndata,type(0.000001));
				if(P[z]>0)
					inc_LCost+=weight*P[z]*log(P[z]/ndata);

				//Advance one channel
				tdata+=width*height;
			}
			*DCost+=inc_DCost;
			*LCost+=inc_LCost;
			free(P);
		}
	
}

/*Weak loss backward kernel*/
template<typename type> static void
weakloss_backward_kernel_cpu(type* derData,
		const type* data,
		const type* pcoords,
		const type* labels,
		type* lambda,
		const type* derOutput,
		const type* A,
		const type* b,
		const type* beta,
		const float limLambda,
		const int pooledIndex,
		const int height,
		const int width,
		const int channels,
		const int numIm)
{


	
	
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

			const type* tdata;
			type *P=(type*)malloc(channels*sizeof(type));

			//Get maximum values to limit the inputs
			type maxLambda=0;
			type maxValue=0;
			type weight=0;
			tdata=data;
			for (int z = 0; z < channels; ++z) {
				l=(int)labels[z];
				maxLambda=std::max(*tdata+A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1],maxLambda);
				maxValue=std::max(maxValue,*tdata);
				tdata+=width*height;
				if(l>0)
					weight+=1-beta[z];
				else
					weight+=beta[z];
			}
			weight=2*weight/channels;
			
			
			//Channels loop
			type sumData=0;
			type normalizer=0;
			type ndata;
			tdata=data;
			for (int z = 0; z < channels; ++z) {
				l=(int)labels[z];
				//Regular case: count over all pixels
				if(l<3){
					P[z]=exp(*tdata+A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1]-maxLambda);
				}
				//Spatially constrained case: consideronly pixels in the boundary
				else{
					if(*pcoords>=RBorders)
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
					P[z]=type(0);
				}
				if(sumData>0)
					ndata=exp(*tdata-maxValue)/sumData;
				else{
					ndata=type(0);
				}

				*derData=weight*derOutput[0]*(ndata-P[z]);
				derData+=width*height;
				tdata+=width*height;
			}
			free(P);
		}
	
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct weakloss<vl::VLDT_CPU, type>
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
      int K=2*channels,Niter=300,*validPixels,*validOuterPixels;
      type *DCost,*DCost_ant,*BCost,*LCost_im;
      type *nlambda,*best_lambda;//,*cpu_labels;
      type mu=10.0,TDCost,TDCost_ant;
      int contFinal=0;
      bool *done;
      int pooledVolume;

      nlambda=(type*)malloc(K*numIm*sizeof(type));
      validPixels=(int *)malloc(numIm*sizeof(int));
      validOuterPixels=(int *)malloc(numIm*sizeof(int));
      DCost_ant=(type*)malloc(numIm*sizeof(type));
      DCost=(type*)malloc(numIm*sizeof(type));
      LCost_im=(type*)malloc(numIm*sizeof(type));
      done=(bool*)calloc(numIm,sizeof(bool));
      BCost=(type*)malloc(numIm*sizeof(type));
      best_lambda=(type*)malloc(K*numIm*sizeof(type));

      //Count number of valid data
      pooledVolume=(int)numIm;
      for (int i=0;i<numIm;i++)
          contValid_kernel_cpu<type>(pcoords,validPixels,validOuterPixels,i, height, width, channels, numIm);

      //Limit lambda values
      for (int i=0;i<numIm*K;i++)
    	  limLambda_kernel_cpu<type>(lambda,labels,A,maxLambda,channels,numIm,i);

      memcpy(best_lambda, lambda, K*numIm*sizeof(type));

      //The loop of iterations has to be here as it is operated serially

      for(iter=0;iter<Niter;iter++){

    	  memcpy(nlambda, lambda, K*numIm*sizeof(type));
    	  //we reset the partial costs
    	  memset(LCost_im,0,numIm*sizeof(type));
    	  memset(DCost,0,numIm*sizeof(type));

    	  //Forward step
    	  for (int i=0;i<numIm*height*width*K;i++)
    		  weakloss_kernel_cpu<type>(LCost_im, DCost, data, pcoords,
    				  labels,lambda,nlambda, A,b,beta,validPixels,validOuterPixels,mu,iter,done,
					  i,height, width, channels, numIm);


    	  //Limit the values of lambda
    	  for (int i=0;i<numIm*K;i++)
    		  limLambda_kernel_cpu<type>(nlambda,labels,A,maxLambda,channels,numIm,i);

    	  contFinal=0;
    	  TDCost=0;
	  bool worse=false;
    	  //Documents loop to update lambdas
    	  for (int d=0;d<numIm;d++){
    		  type mejora;

    		  if(iter==0){
    			  mejora=100.0;
    			  TDCost+=DCost[d];
    		  }
    		  else if(done[d]==true){
    			  mejora=0;
    			  TDCost+=DCost_ant[d];
    		  }
    		  else{
    			  mejora=100.0*(DCost[d]-DCost_ant[d])/abs(DCost_ant[d]);
    			  TDCost+=DCost[d];
    		  }
		//Lambda updates
    		  //If we worse the results
    		  if(mejora<-0.1){
    		  	 done[d]=false;
    		  	 //We set the previous best lambda
    		  	 memcpy(lambda+K*d,best_lambda+K*d, K*sizeof(type));
    		  	 worse=true;
    		  }
    		  else if(mejora<=0.01 ){
    			  done[d]=true;
    		  }
    		  //if we are improving
    		  else{
    			  //Copy the best option (only in the respective locations)
    			   memcpy(best_lambda+K*d, lambda+K*d, K*sizeof(type));
    			  //Copy nlambda to lambda
    			  memcpy(lambda+K*d, nlambda+K*d, K*sizeof(type));
    			  DCost_ant[d]=DCost[d];
    			  BCost[d]=LCost_im[d];
			  done[d]=false;
    		  }
    	  }

 	if(worse && mu>1e-2){
    		  mu=mu*0.5;
    		  //printf("iter %d cost %f mu %f\n",iter,TDCost,mu);
    	  }
    	  else{
    		  type mejora=100.0*(TDCost-TDCost_ant)/abs(TDCost_ant);
    		 // printf("iter %d cost %f mejora %f mu %f\n",iter,TDCost,mejora,mu);
    		  //After 3 iters without improving
    		  if(mejora<0.001 && iter>4){
    			  //printf("iter %d cost %f mejora %f mu %f\n",iter,TDCost,mejora,mu);
    			  break;
    		  }
    		  TDCost_ant=TDCost;
    	  } 
      }

      type TBCost=0;
      for (int d=0;d<numIm;d++){
    	  TBCost+=BCost[d];
      }
      *LCost=TBCost;
      memcpy(lambda, best_lambda, K*numIm*sizeof(type));
      free(nlambda);
      free(validPixels);
      free(DCost);
      free(best_lambda);
      return VLE_Success ;

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

    	//Backward kernel
    	for (int i=0;i<numIm*height*width;i++)
    		weakloss_backward_kernel_cpu<type>(derData, data, pcoords, labels, lambda, derOutput, A, b, beta,maxLambda,
				i,height, width, channels, numIm);



    	return vl::VLE_Success;
    }
  } ; // weakloss


} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::weakloss<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::weakloss<vl::VLDT_CPU, double> ;
#endif

