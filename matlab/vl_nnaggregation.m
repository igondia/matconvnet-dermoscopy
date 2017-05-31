% VL_NNAGGREGATION  CNN aggregation
%    Y = VL_NNPOOL(X, MASK) applies the pooling operator to all
%    channels of the data X only in locations marked by MASK and with r parameter. X is a
%    SINGLE array of dimension H x W x D x N where (H,W) are the
%    height and width of the map stack, D is the image depth (number
%    of feature channels) and N the number of of images in the stack.
%
%    Y = VL_NNAGGREGATION(X, MASK) uses r.
%
%    DZDX = VL_NNAGGREGATION(X, MASK, DZDY) computes the derivatives of
%    the nework output Z w.r.t. the data X given the derivative DZDY
%    w.r.t the max-pooling output Y.
%
%    VL_NNAGGREGATION(..., 'option', value, ...) takes the following options:
%
%
%    Method:: ['max']
%      Specify method of pooling. It can be either 'max' (retain max value
%      over the pooling region per channel) or 'avg' (compute the average
%      value over the poolling region per channel) or lse.
%
%    r:: [1]
%      r parameter in lse method

%    ## CUDNN SUPPORT
%
%    If compiled in, the function will use cuDNN convolution routines
%    (with the exception of asymmetric left-right or top-bottom
%    padding and avergage pooling that triggers a bug in cuDNN). You
%    can use the 'NoCuDNN' option to disable cuDNN or 'cuDNN' to
%    activate it back again (the choice sticks until MATLAB purges the
%    MEX files for any reason).


