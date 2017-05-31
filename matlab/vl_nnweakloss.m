% VL_NNWEAKLOSS  CNN function that implements the loss used for weak
% segmentation
%    [Y,lambda] = VL_NNWEAKLOSS(X, MASK, LABELS, LAMBDA) applies the weak constraints defined
%    by the labels and the masks
%    
%    [DZDX, lambda] = VL_NNWEAKLOSS(X, MASK, LABELS, LAMBDA, DZDY) computes the derivatives of
%    the nework output Z w.r.t. the data X given the derivative DZDY
%    w.r.t the max-pooling output Y.
%
%    VL_NNWEAKLOSS(..., 'option', value, ...) takes the following options:
%
%
%    lambda:: [1]
%      lambda parameter to impose restrictions

%    ## CUDNN SUPPORT
%
%    If compiled in, the function will use cuDNN convolution routines
%    (with the exception of asymmetric left-right or top-bottom
%    padding and avergage pooling that triggers a bug in cuDNN). You
%    can use the 'NoCuDNN' option to disable cuDNN or 'cuDNN' to
%    activate it back again (the choice sticks until MATLAB purges the
%    MEX files for any reason).


