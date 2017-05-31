function net = vl_simplenn_clean(net)
%VL_SIMPLENN_TIDY  Clean a SimpleNN network to use in testing mode.
%   NET = VL_SIMPLENN_CLEAN(NET) takes the NET object and upgrades
%   it to the current version of MatConvNet. This is necessary in
%   order to allow MatConvNet to evolve, while maintaining the NET
%   objects clean.
%
%   The function is also generally useful to fill in missing default
%   values in NET.
%
%   See also: VL_SIMPLENN().

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% copy layers
for l = 1:numel(net.layers)
  defaults = {'precious', false};
  layer = net.layers{l} ;
    if isfield(layer, 'momentum')
        layer = rmfield(layer, 'momentum') ;
    end
    net.layers{l} = layer;
end
  