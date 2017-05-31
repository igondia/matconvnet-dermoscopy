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
#include "bits/nnaggregation.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_r=0,
  opt_method,
  opt_verbose,
  opt_cudnn,
  opt_no_cudnn,
} ;

/* options */
VLMXOption  options [] = {
  {"r",                1,   opt_r                 },
  {"Method",           1,   opt_method            },
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
  IN_DATA = 0, IN_VALIDOS, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  vl::AggregationMethod method = vl::vlAggregationMax ;
  double r=1;
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 2) {
    mexErrMsgTxt("The arguments are less than two.") ;
  }

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {


    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_method :
        if (!vlmxIsString(optarg,-1)) {
           vlmxError(VLMXE_IllegalArgument, "METHOD is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "max")) {
          method = vl::vlAggregationMax ;
        } else if (vlmxIsEqualToStringI(optarg, "avg")) {
          method = vl::vlAggregationAverage ;
        } else if (vlmxIsEqualToStringI(optarg, "lse")) {
          method = vl::vlAggregationLse ;  
        } else {
          vlmxError(VLMXE_IllegalArgument, "METHOD is not a supported method.") ;
        }
        break;
        
      case opt_r :
    	  if (!vlmxIsPlainScalar(optarg)) {
    		  mexErrMsgTxt("r is not a plain scalar.") ;
    	  }
    	  r = mxGetPr(optarg)[0] ;
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

  vl::MexTensor data(context) ;
  vl::MexTensor validos(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ; // -> 4 dimensions
  validos.init(in[IN_VALIDOS]);
  validos.reshape(4);
  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ; // -> 4 dimensions
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }

  /* Get the output Shape */
  vl::TensorShape outputShape(1,1,data.getDepth(),data.getSize());

  if (backMode && (derOutput != outputShape)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnaggregation: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::VLDT_GPU) ? "VLDT_GPU" : "CPU") ;
    if (data.getDeviceType() == vl::VLDT_GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled() ? "cuDNN" : "MatConvNet") ;
#else
      mexPrintf("; MatConvNet\n") ;
#endif
    } else {
      mexPrintf("; MatConvNet\n") ;
    }
    vl::print("vl_nnaggregation: data: ", data) ;
    mexPrintf("vl_nnaggregation: method: %s\n", (method == vl::vlAggregationMax) ? "max" : (method == vl::vlAggregationLse) ? "lse" : "avg") ;
    if (backMode) {
      vl::print("vl_nnaggregation:: derOutput: ", derOutput) ;
      vl::print("vl_nnaggregation:: derData: ", derData) ;
    } else {
      vl::print("vl_nnaggregation:: output: ", output) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  if (!backMode) {
	//mexPrintf("forward vl_nnaggregation: pooling: %d x %d\n", poolRings, poolAngles);
    error = vl::nnaggregation_forward(context,
                                  output, data,
                                  validos,
                                  method,
                                  r) ;
  } else {
	//mexPrintf("backward vl_nnaggregation: pooling: %d x %d\n", poolRings, poolAngles);
    error = vl::nnaggregation_backward(context,
                                   derData, data, validos,
                                   derOutput,
                                   method,
                                   r) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
