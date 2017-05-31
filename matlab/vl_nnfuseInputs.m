function [y] = vl_nnfuseInputs(x,wmod,varargin)
%VL_NNDROPOUT CNN dropout.
%   [Y] = VL_NNFUSEINPUTS(RES,LAYERS,SIZEOF) fuse inputs of layers indicated 
%   by LAYERS at the size of the level SIZEOF
%
%   [DZDX] = VL_NNFUSEINPUTS((RES, LAYERS,SIZEOF, DZDY, ) computes the
%   derivatives of the blocks projected onto DZDY. 

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end
gpuMode = isa(x, 'gpuArray');

modChannels=size(wmod,3);

if ~backMode
    oSize=size(x);
    iChannels=size(x,3);
    outChannels=iChannels+modChannels;
    oSize(3)=outChannels;
    if(gpuMode)
%         y=gpuArray(zeros(oSize,'single'));
    	mod=gpuArray(imresize(gather(wmod),[oSize(1) oSize(2)],'nearest'));
    else
%         y=zeros(oSize,'single');
        mod=imresize(wmod,[oSize(1) oSize(2)],'nearest');
    end
    y=cat(3,x,mod);
%BackMode: We have to select the corresponding layer    
else
    
    iChannels=size(x,3);
    y=dzdy(:,:,1:iChannels,:);
    
end

% numLayers=length(layers);
% mainLayer=layers(mainLayerIdx);
% if ~backMode
%     y=[];
%     oSize=size(res(mainLayer).x);
%     for lidx=1:numLayers
%         l=layers(lidx);
%         if(size(res(l).x,1)~=oSize(1) || size(res(l).x,2)~=oSize(2))
%             if(gpuMode)
%                 x=gpuArray(imresize(gather(res(l).x),[oSize(1) oSize(2)]));
%             else
%                 x=imresize(res(l).x,[oSize(1) oSize(2)]);
%             end
%         else
%             x=res(l).x;
%         end
%         x=x*fusionWeights(lidx);
%         if(isempty(y))
%             y=x;
%         else
%             y=cat(3,y,x);
%         end
%     end
% %BackMode: We have to select the corresponding layer    
% else
%     contIdx=0;
%     lastLayer=max(layers);
%     %The selected layer is the one with largest index
%     for lidx=1:numLayers
%         l=layers(lidx);
%         if(l==lastLayer)
%             y=dzdy(:,:,contIdx+1:contIdx+size(res(lastLayer).x,3),:);
%             break;
%             if(gpuMode)
%                 y=gpuArray(imresize(y,[size(res(lastLayer).x,1) size(res(lastLayer).x,2)]));
%             else
%                 y=imresize(y,[size(res(lastLayer).x,1) size(res(lastLayer).x,2)]);
%             end
%         else
%             contIdx=contIdx+size(res(l).x,3);
%         end
%     end
%     
% end