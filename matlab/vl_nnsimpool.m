%VL_NNSIMPOOL CNN simmetry poolinng.
%   Y = VL_NNSIMPOOL(X) applies the pooling operator to all
%   channels of the data X using a square filter of size POOL. X is a
%   SINGLE array of dimension H x W x D x N where (H,W) are the
%   height and width of the map stack, D is the image depth (number
%   of feature channels) and N the number of of images in the stack.
%
%   Y = VL_NNSIMPOOL(X) uses a rectangular filter of
%   height POOLY and width POOLX.
%
%   DZDX = VL_NNSIMPOOL(X, DZDY) computes the derivatives of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.
%
%   VL_NNCONV(..., 'option', value, ...) takes the following options:
%
%  
%   ## CUDNN SUPPORT
%
%   If compiled in, the function will use cuDNN convolution routines
%   (with the exception of asymmetric left-right or top-bottom
%   padding and avergage pooling that triggers a bug in cuDNN). You
%   can use the 'NoCuDNN' option to disable cuDNN or 'cuDNN' to
%   activate it back again (the choice sticks until MATLAB purges the
%   MEX files for any reason).

% Copyright (C) 2014 Andrea Vedaldi, Karel Lenc, and Max Jaderberg.
% Copyright (C) 2015 Andrea Vedaldi and Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

