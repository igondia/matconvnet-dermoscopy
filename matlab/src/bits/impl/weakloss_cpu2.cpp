// @file pooling_cpu.cpp
// @brief Pooling block implementation (VLDT_GPU)
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "weakloss.hpp"
#include "../data.hpp"
#include <algorithm>
#include <limits>
#include <math.h>
#include "../mexutils.h"

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


template<typename type> static inline void
weakloss_forward_cpu(type* cost,
                    type const* data,
                    type const* pcoords,
                    type const* labels,
                    type * lambda,
                    type const* A,
                    type const* b,
                    type const* beta,
                    float const maxValLambda,
                    size_t height, size_t width, size_t channels, size_t numIm)
{

	int K=2*channels,l,contPatterns;
	type ndata,maxLambda,maxValue;
	int validPixels;
	int Niter=1000;
	type P[channels];
	type normalizer=0;
	type sumData=0,limLambda=10.0;
	type reg[K];
	type mu=1.00;

	const type *tdata;
	//Pointers to original locations for various iterations
	const type *odata=data;
	const type *opcoords=pcoords;
	const type *olabels=labels;
	type *olambda=lambda;
	double DCost_ant=0,DCost,DCost1,DCost2,RegCost,LCost=0;

	//Iterations loop
	for(int iter=0;iter<Niter;iter++){

		//Reset pointers and cost
		data=odata;
		pcoords=opcoords;
		labels=olabels;
		lambda=olambda;
		DCost1=0;
		DCost2=0;
		DCost=0;
		LCost=0;
		RegCost=0;

		//Images loop
		for (int i = 0; i < numIm ; ++i) {


			//Limit lambda to a maximum value and get the max lambda to limit the inputs to improve numeric stability
			maxLambda=0.0;
			for (int z = 0; z < channels; ++z) {
				//Limit lambda
				lambda[2*z]=std::max(lambda[2*z],0);
				lambda[2*z+1]=std::min(lambda[2*z+1],maxValLambda);
				//Get max lambda
				maxLambda=std::max(lambda[2*z],maxLambda);
				maxLambda=std::max(lambda[2*z+1],maxLambda);
			}

			//Counter for valid pixels
			validPixels=0;
			memset(reg, 0, K*sizeof(type));
			//Pixels Loop
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					//If the pixel is valid
					if(pcoords[x * height + y]>=0){
						validPixels++;

						tdata=data;
						maxValue=-1000;
						for (int z = 0; z < channels; ++z) {
							maxValue=std::max(maxValue,tdata[x * height +y]);
							tdata+=width*height;
						}

						//Channels loop
						sumData=0;
						normalizer=0;
						tdata=data;
						for (int z = 0; z < channels; ++z) {
							l=(int)labels[z];
							if(l!=3){
								P[z]=exp(tdata[x * height +y] +A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1]-maxValue-maxLambda);
								if(isnan(P[z]) || isinf(P[z])){
									//mexPrintf("Error NaN forward P(z): %f tdata: %f reg %f\n",P[z],tdata[x * height +y],A[2*l]*lambda[2*z]-A[2*l+1]*lambda[2*z+1]);mexEvalString("drawnow");
									mexPrintf("Error NaN forward iter %d im %d x,y=%d,%d l=%d tdata: %f A1 %f lambda1 %f A2 %f lambda2 %f reg %f norm %f\n",iter,i,x,y,l,tdata[x * height +y],A[2*l],lambda[2*z],A[2*l+1],lambda[2*z+1],A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1],maxValue+maxLambda);mexEvalString("drawnow");
									mexErrMsgTxt("Error\n");
								}
							}
							//Spatial case, we have to either apply one or the other constraint depending
							//on the pixel location
							else{
								if(pcoords[x * height + y]>=0.8)
									P[z]=exp(tdata[x * height +y]+A[2*l]*lambda[2*z]-maxValue-maxLambda);
								else
									P[z]=exp(tdata[x * height +y]+A[2*l+1]*lambda[2*z+1]-maxValue-maxLambda);
							}
							normalizer+=P[z];
							sumData+=exp(tdata[x * height +y]-maxValue);
							//Advance one channel
							tdata+=width*height;
						}

						//Normalization and Lagrangian update
						tdata=data;
						for (int z = 0; z < channels; ++z){

							if(normalizer>0){
								P[z]=P[z]/normalizer;
								//Dual Cost
								DCost1-=log(normalizer)+maxValue+maxLambda;
							}
							else{

								P[z]=1.0/channels;
								mexErrMsgTxt("Raro a cero\n");
							}
							if(sumData>0)
								ndata=exp(tdata[x * height +y]-maxValue)/sumData;
							else
								ndata=1.0/channels;

							l=(int)labels[z];
							if(isnan(P[z])){
								mexPrintf("Error NaN forward P(z): %f normalizer %f \n",P[z],normalizer);mexEvalString("drawnow");
								mexErrMsgTxt("Error\n");
							}
							if(l!=3){
								reg[2*z]+=A[2*l]*P[z];
								reg[2*z+1]+=A[2*l+1]*P[z];
							}
							else{
								if(pcoords[x * height + y]>=0.8)
									reg[2*z]+=A[2*l]*P[z];
								else
									reg[2*z+1]+=A[2*l+1]*P[z];
							}
							//Update Lagrangian Cost
							if(P[z]>0.000001)
								LCost+=P[z]*(log(P[z])-log(ndata));
							if(isinf(LCost)){
								mexPrintf("P[z] %f reg %f ndata %f tdata %f maxValues %f sumData %f\n",P[z],A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1],ndata,tdata[x * height +y],maxValue+maxLambda,sumData);mexEvalString("drawnow");
								mexErrMsgTxt("Error\n");
							}
						}

					}//Valid location


				}//X loop
			}//Y loop
			//mexPrintf("Partial LCost: %f\n",LCost);mexEvalString("drawnow");

			//Now, we need to compute the final regularization term
			for(int z=0;z<channels;z++){
				l=(int)labels[z];
				//First constraint

				//Dual Cost
				DCost2+=lambda[2*z]*b[2*l]*validPixels;
				type inc=b[2*l]*validPixels-reg[2*z];
				//Lagrangian Cost
				LCost+=lambda[2*z]*inc;
				//Update lambda
				//mexPrintf("Lambda 1: z %d label %d lambda %f b: %f AP %f\n",z,l,lambda[2*z], b[2*l]*validPixels,reg[2*z] );mexEvalString("drawnow");
				lambda[2*z]=lambda[2*z]+(mu/validPixels)*inc;
				lambda[2*z]=std::max(lambda[2*z],type(0));
				lambda[2*z]=std::min(lambda[2*z],beta[z]*maxValLambda);
				if(isinf(LCost)){
					mexPrintf("Lambda 1 %f inc %f\n",lambda[2*z],inc);mexEvalString("drawnow");
					mexErrMsgTxt("Error\n");
				}
				//mexPrintf("Lambda 1 d: %f\n",lambda[2*z]);mexEvalString("drawnow");

				//Second constraint
				//Dual Cost
				DCost2+=lambda[2*z+1]*b[2*l+1]*validPixels;
				inc=b[2*l+1]*validPixels-reg[2*z+1];
				//Lagrangian Cost
				LCost+=lambda[2*z+1]*inc;
				//Update lambda
				//mexPrintf("Lambda 2: z %d label %d %f b: %f AP %f\n",z,l,lambda[2*z+1], b[2*l+1]*validPixels,reg[2*z+1] );mexEvalString("drawnow");
				lambda[2*z+1]=lambda[2*z+1]+(mu/validPixels)*inc;
				lambda[2*z+1]=std::max(lambda[2*z+1],type(0));
				lambda[2*z+1]=std::min(lambda[2*z+1],beta[z]*maxValLambda);
				//Projected Gradient Ascent to update lambda

				if(isinf(LCost)){
					mexPrintf("Lambda 2 %f inc %f\n",lambda[2*z+1],inc);mexEvalString("drawnow");
					mexErrMsgTxt("Error\n");
				}

			}

			//Let's advance in arrays one image
			lambda+=2*channels;
			//We advance one image on the data
			data+=width*height*channels;
			pcoords+=2*width*height;
			//We advance one image in the labels
			labels+=channels;
		}//Image Loop
		DCost=DCost1+DCost2;
		double mejora=100.0*(DCost-DCost_ant)/abs(DCost_ant);

		//mexPrintf("Iter %d DCost1 %f DCost2 %f DCost: %f improv %lf\n",iter,DCost1,DCost2,DCost,mejora);mexEvalString("drawnow");

		if(mejora<0.01 && iter>0)
			break;
		else{
			//double mejora=100*(DCost-DCost_ant)/abs(DCost_ant);
			DCost_ant=DCost;
		}

	}
	*cost=LCost;
}


/* ---------------------------------------------------------------- */
/*                                               pooling_*_backward */
/* ---------------------------------------------------------------- */

/*
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */

template<typename type> static inline void
weakloss_backward_cpu(type* derData,
                    type const* data,
                    type const* pcoords,
                    type const* labels,
                    type * lambda,
                    type const* derOutput,
                    type const* A,
                    type const* b,
                    size_t height, size_t width, size_t channels, size_t numIm)
{

	int K=2*channels,l;
	type LCost=0,ndata,maxValue,maxLambda;
	int validPixels;
	type P[channels];
				//NP    //LP      //GP    //OP
	//type A[8]={-1, 0,    1,  -1,    1, 0,   1,-1};
	//type b[8]={ 0, 0,   0.1, -0.5,   0.5, 0, 0.1, 0};

	type reg[K];
	type mu=0.0;

	const type *tdata;
	type *tderData;


	//Images loop
	for (int i = 0; i < numIm ; ++i) {

		//Get maximum values to limit the inputs
		maxLambda=0.0;
		for (int z = 0; z < channels; ++z) {
			maxLambda=std::max(lambda[2*z],maxLambda);
			maxLambda=std::max(lambda[2*z+1],maxLambda);
		}

		//Initializations
		validPixels=0;
		//memset (reg, 0, sizeof(reg));
		//mexPrintf("IMagen %d\n",i);
		//Pixels Loop
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				if(pcoords[x * height + y]>=0){
					validPixels++;
				}
			}
		}
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {

				//If they are valid
				if(pcoords[x * height + y]>=0){

					//Getting normalizers
					tdata=data;
					maxValue=-1000;
					for (int z = 0; z < channels; ++z) {
						maxValue=std::max(maxValue,tdata[x * height +y]);
						tdata+=width*height;
					}

					//We initialize tdata
					tdata=data;
					type normalizer=0;
					type sumData=0;
					//Channels loop
					for (int z = 0; z < channels; ++z) {
						l=(int)labels[z];
						if(l!=3){
							P[z]=exp(tdata[x * height +y] +A[2*l]*lambda[2*z]+A[2*l+1]*lambda[2*z+1]-maxValue-maxLambda);
							if(isnan(P[z])){
								mexPrintf("Error NaN backward P(z): %f \n",P[z]);mexEvalString("drawnow");

							}
							//if(x==5 && y==5 && lambda[2*z]==0 && lambda[2*z+1]==0)
							//	mexPrintf("A ceros z %d l %d P(z) %f ndata %f\n",z,l,P[z],exp(tdata[x * height +y]));
						}
						//Spatial case, we have to either apply one or the other constraint depending
						//on the pixel location
						else{
							if(pcoords[x * height + y]>=0.8)
								P[z]=exp(tdata[x * height +y] +A[2*l]*lambda[2*z]-maxValue-maxLambda);
							else
								P[z]=exp(tdata[x * height +y] +A[2*l+1]*lambda[2*z+1]-maxValue-maxLambda);
						}
						normalizer+=P[z];
						sumData+=exp(tdata[x * height +y]-maxValue-maxLambda);
						tdata+=width*height;
					}
					//sumData=std::max(sumData,type(0.0000001));
					//normalizer=std::max(sumData,type(0.0000001));

					//reset tdata location
					tdata=data;
					tderData=derData;
					//Normalization and Lagrangian update
					for (int z = 0; z < channels; ++z){

						if(normalizer>0)
							P[z]=P[z]/normalizer;
						else{
							P[z]=1.0/channels;
							mexPrintf("Raro\n");
						}
						if(sumData>0)
							ndata=exp(tdata[x * height +y]-maxValue-maxLambda)/sumData;
						else{
							ndata=1.0/channels;
							mexPrintf("Raro\n");
						}
						if(isnan(P[z])){
							mexPrintf("Error NaN P(z): %f normalizer %f\n",P[z],normalizer);mexEvalString("drawnow");

						}
						tderData[x * height +y]=derOutput[0]*(ndata-P[z]);

						//Move arrays
						tdata+=width*height;
						tderData+=width*height;
					}
				}//Valid location
				else{
					tderData=derData;
					for (int z = 0; z < channels; ++z){
						tderData[x * height +y]=0;
						tderData+=width*height;
					}
				}

			}//X loop
		}//Y loop


		//Let's advance in arrays
		lambda+=2*channels;
		//We advance one image on the data
		data+=width*height*channels;
		derData+=width*height*channels;
		pcoords+=2*width*height;
		//We advance one image in the labels
		labels+=channels;
	}//Image Loop

}



/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct weakloss<vl::VLDT_CPU, type>
  {
    static vl::ErrorCode
    forward(type* cost,
            type const* data,
            type const* pcoords,
            type const* labels,
            type * lambda,
            type const* A,
            type const* b,
            type const* beta,
            float const maxLambda,
            size_t height, size_t width, size_t channels, size_t numIm)
    {
      weakloss_forward_cpu<type>  (cost,
                                   data,
                                   pcoords,
                                   labels,
                                   lambda,
                                   A,
                                   b,
                                   beta,
                                   maxLambda,
                                   height, width, channels, numIm) ;
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
    	weakloss_backward_cpu<type> (derData,
                                     data,
                                     pcoords,
                                     labels,
                                     lambda,
                                     derOutput,
                                     A,
                                     b,
                                     height, width, channels, numIm) ;
      return VLE_Success ;
    }
  }; // weakloss
} };



// Instantiations
template struct vl::impl::weakloss<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::weakloss<vl::VLDT_CPU, double> ;
#endif

