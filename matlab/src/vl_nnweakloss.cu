// @file vl_nnpool.cu
// @brief Pooling block MEX wrapper
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnweakloss.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_verbose=0,
  opt_cudnn,
  opt_no_cudnn,
  opt_maxLambda,
} ;

/* options */
VLMXOption  options [] = {
  {"maxLambda",       10,   opt_maxLambda         },
  {"Verbose",          1,   opt_verbose           },
  {"CUDNN",            0,   opt_cudnn             },
  {"NoCUDNN",          0,   opt_no_cudnn          },
  {0,                  0,   0                     }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_PCOORDS, IN_LABELS, IN_LAMBDA, IN_A, IN_b, IN_beta, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_LAMBDA, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool backMode = false ;
  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg;
  mxArray *auxLambda ;
  float maxLambda=10;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 7) {
    mexErrMsgTxt("The arguments are less than SEVEN.") ;
  }
  //Si hay más de 7 parámetros y el octavo es un string es forward
  if (nin > 7 && vlmxIsString(in[7],-1)) {
    next = 7 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 8) ;
  }


  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {


    switch (opt) {
    case opt_maxLambda :
        	  if (!vlmxIsPlainScalar(optarg)) {
        		  mexErrMsgTxt("maxLambda is not a plain scalar.") ;
        	  }
        	  maxLambda = (float)mxGetPr(optarg)[0] ;
        	  break ;

    case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_no_cudnn :
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(false) ;
#endif
        break ;

      case opt_cudnn :
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(true) ;
#endif
        break ;


    	  
      default:
        break ;
    }
  }

  vl::MexTensor data(context);
  vl::MexTensor pcoords(context);
  vl::MexTensor labels(context);
  vl::MexTensor lambda(context);
  vl::MexTensor A(context);
  vl::MexTensor b(context);
  vl::MexTensor beta(context);
  vl::MexTensor derOutput(context);
  //vl::MexTensor lambdaOutput(context);


  data.init(in[IN_DATA]);
  data.reshape(4) ; // -> 4 dimensions
  pcoords.init(in[IN_PCOORDS]);
  pcoords.reshape(4);
  labels.init(in[IN_LABELS]);
  labels.reshape(2);
  A.init(in[IN_A]);
  //A.reshape(2);
  b.init(in[IN_b]);
  //b.reshape(2);
  beta.init(in[IN_beta]);
  beta.reshape(2);
  
  //Porque viene const
  auxLambda=mxDuplicateArray(in[IN_LAMBDA]);
  lambda.init(auxLambda);
  lambda.reshape(2);


  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ; // -> 4 dimensions
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }

  /* The output shape is just a number*/
  vl::TensorShape outputShape(1,1,1,1);

  if (backMode && (derOutput != outputShape)) {
	  mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType();
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context);
  vl::MexTensor derData(context);


  if (!backMode) {

    output.initWithZeros(deviceType, dataType, outputShape) ;

  } else {
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
  }

  if (verbosity > 0) {
    //mexPrintf("vl_nnweakloss: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::VLDT_GPU) ? "VLDT_GPU" : "CPU") ;
    if (data.getDeviceType() == vl::VLDT_GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled() ? "cuDNN" : "MatConvNet") ;
#else
      mexPrintf("; MatConvNet\n") ;
#endif
    } else {
      mexPrintf("; MatConvNet\n") ;
    }
    vl::print("vl_nnweakloss: data: ", data) ;
    if (backMode) {
      vl::print("vl_nnweakloss:: derOutput: ", derOutput) ;
      vl::print("vl_nnweakloss:: derData: ", derData) ;
    } else {
      vl::print("vl_nnweakloss:: output: ", output) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  if (!backMode) {

    error = vl::nnweakloss_forward(context,
                                  output,
                                  data,
                                  pcoords,
                                  labels,
                                  lambda,
                                  A,
                                  b,
                                  beta,
                                  maxLambda) ;
  } else {
    error = vl::nnweakloss_backward(context,
                                   derData,
                                   data,
                                   pcoords,
                                   labels,
                                   lambda,
                                   derOutput,
                                   A,
                                   b,
                                   beta,
                                   maxLambda);
  }
  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */
  
  if (error != vl::VLE_Success) {
	  mexPrintf("Hay error\n");
    vlmxError(VLMXE_IllegalArgument, context.getLastErrorMessage().c_str()) ;
  }

  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
  out[OUT_LAMBDA] = lambda.relinquish() ;
}
