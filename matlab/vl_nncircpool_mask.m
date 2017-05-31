% VL_CIRCNNPOOL_MASK  CNN pooling
%    Y = VL_CIRCNNPOOL_MASK(X, PCOORDS, [RINGS, ANGLES]) applies the polar pooling operator to all
%    channels of the data X using RINGS and ANGLES. X is a
%    SINGLE array of dimension H x W x D x N where (H,W) are the
%    height and width of the map stack, D is the image depth (number
%    of feature channels) and N the number of of images in the stack.
%
%    DZDX = VL_CIRCNNPOOL_MASK(X, PCOORDS, [RINGS, ANGLES], DZDY) computes the derivatives of
%    the nework output Z w.r.t. the data X given the derivative DZDY
%    w.r.t the max-pooling output Y.
%
%    VL_CIRCNNPOOL_MASK(..., 'option', value, ...) takes the following options:
%
%    Overlap:: [0]
%      The overlap [OVERLAP_RING OVERLAP_ANGLE] used in rings and angles, so that pixels may belong
%      to more than one sector.
%
%    Pad:: [0]
%      The amount of input padding. Input images are padded with zeros
%      by this number of pixels on all sides before the convolution is
%      computed. It can also be a vector [TOP BOTTOM LEFT RIGHT] to
%      specify a different amount of padding in each direction. The
%      size of the poolin filter has to exceed the padding.
%
%    Method:: ['max']
%      Specify method of pooling. It can be either 'max' (retain max value
%      over the pooling region per channel) or 'avg' (compute the average
%      value over the poolling region per channel).
%
%    The derivative DZDY has the same dimension of the output Y and
%    the derivative DZDX has the same dimension as the input X.
%



