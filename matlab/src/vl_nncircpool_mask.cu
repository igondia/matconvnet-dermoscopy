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
#include "bits/nncircpooling_mask.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_overlap = 0,
  opt_pad,
  opt_method,
  opt_verbose,
  opt_cudnn,
  opt_no_cudnn,
} ;

/* options */
VLMXOption  options [] = {
  {"Overlap",           1,   opt_overlap            },
  {"Pad",              1,   opt_pad               },
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
  IN_DATA = 0, IN_COORDS, IN_SIZE, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int poolAngles ;
  int poolRings ;
  float overlapRing = 0 ;
  float overlapAngle = 0 ;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;
  vl::PoolingMethod method = vl::vlCircPoolingMaskMax ;
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 3) {
    mexErrMsgTxt("The arguments are less than three.") ;
  }

  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {


    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_overlap :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("STRIDE is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
        	  overlapRing = (float)mxGetPr(optarg)[0] ;
        	  overlapAngle = overlapRing ;
            break ;
          case 2:
        	  overlapRing = (float)mxGetPr(optarg)[0] ;
        	  overlapAngle = (float)mxGetPr(optarg)[1] ;
            break ;
          default:
            mexErrMsgTxt("OVERLAP has neither one nor two elements.") ;
        }
        break ;

      case opt_pad :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("PAD is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            padLeft = (int)mxGetPr(optarg)[0] ;
            padRight = padLeft ;
            padTop = padLeft ;
            padBottom = padLeft ;
            break ;
          case 4:
            padTop = (int)mxGetPr(optarg)[0] ;
            padBottom = (int)mxGetPr(optarg)[1] ;
            padLeft = (int)mxGetPr(optarg)[2] ;
            padRight = (int)mxGetPr(optarg)[3] ;
            break ;
          default:
            mexErrMsgTxt("PAD has neither one nor four elements.") ;
        }
        break;

      case opt_method :
        if (!vlmxIsString(optarg,-1)) {
           vlmxError(VLMXE_IllegalArgument, "METHOD is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "max")) {
          method = vl::vlCircPoolingMaskMax ;
        } else if (vlmxIsEqualToStringI(optarg, "avg")) {
          method = vl::vlCircPoolingMaskAverage ;
        } else {
          vlmxError(VLMXE_IllegalArgument, "METHOD is not a supported method.") ;
        }
        break;

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
  vl::MexTensor pcoords(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ; // -> 4 dimensions
  pcoords.init(in[IN_COORDS]);
  pcoords.reshape(4);
  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ; // -> 4 dimensions
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }

  if (!vlmxIsPlainMatrix(in[IN_SIZE],-1,-1)) {
    mexErrMsgTxt("SIZE is not a plain matrix.") ;
  }
  switch (mxGetNumberOfElements(in[IN_SIZE])) {
    case 1:
      poolRings = mxGetPr(in[IN_SIZE])[0] ;
      poolAngles = 1;
      break ;
    case 2:
      poolRings = mxGetPr(in[IN_SIZE])[0] ;
      poolAngles = mxGetPr(in[IN_SIZE])[1] ;
      break ;
    default:
      mexErrMsgTxt("SIZE has neither one nor two elements.") ;
  }

  if (poolRings == 0 || poolAngles == 0) {
    mexErrMsgTxt("A dimension of the pooling SIZE is void.") ;
  }
 /* if (data.getHeight() + (padTop+padBottom) < poolHeight ||
      data.getWidth() + (padLeft+padRight) < poolWidth) {
    mexErrMsgTxt("The pooling window is larger than the DATA (including padding).") ;
  }
  if (padLeft < 0 ||
      padRight < 0 ||
      padTop < 0 ||
      padBottom < 0) {
    mexErrMsgTxt("An element of PAD is negative.") ;
  }*/
  /*if (padLeft >= poolWidth ||
      padRight >= poolWidth ||
      padTop >= poolHeight  ||
      padBottom >= poolHeight) {
    mexErrMsgTxt("A padding value is larger or equal than the size of the pooling window.") ;
  }*/

  /* Get the output Shape */
  vl::TensorShape outputShape(poolRings,poolAngles,data.getDepth(),data.getSize());

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
    mexPrintf("vl_nncircpoolmask: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::VLDT_GPU) ? "VLDT_GPU" : "CPU") ;
    if (data.getDeviceType() == vl::VLDT_GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled() ? "cuDNN" : "MatConvNet") ;
#else
      mexPrintf("; MatConvNet\n") ;
#endif
    } else {
      mexPrintf("; MatConvNet\n") ;
    }
    mexPrintf("vl_nncircpoolmask: overlap: [%d %d], pad: [%d %d %d %d]\n",
              overlapRing, overlapAngle,
              padTop, padBottom, padLeft, padRight) ;
    vl::print("vl_nncircpoolmask: data: ", data) ;
    mexPrintf("vl_nncircpoolmask: pooling: %d x %d\n", poolRings, poolAngles);
    mexPrintf("vl_nncircpoolmask: method: %s\n", (method == vl::vlCircPoolingMaskMax) ? "max" : "avg") ;
    if (backMode) {
      vl::print("vl_nncircpoolmask:: derOutput: ", derOutput) ;
      vl::print("vl_nncircpoolmask:: derData: ", derData) ;
    } else {
      vl::print("vl_nncircpoolmask:: output: ", output) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  if (!backMode) {
	//mexPrintf("forward vl_nncircpoolmask: pooling: %d x %d\n", poolRings, poolAngles);
    error = vl::nncircpooling_mask_forward(context,
                                  output, data,
                                  pcoords,
                                  method,
                                  poolRings,poolAngles,
								  overlapRing, overlapAngle,
                                  padTop, padBottom, padLeft, padRight) ;
  } else {
	//mexPrintf("backward vl_nncircpoolmask: pooling: %d x %d\n", poolRings, poolAngles);
    error = vl::nncircpooling_mask_backward(context,
                                   derData, data, pcoords,
                                   derOutput,
                                   method,
                                   poolRings,poolAngles,
								   overlapRing, overlapAngle,
                                   padTop, padBottom, padLeft, padRight) ;
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
