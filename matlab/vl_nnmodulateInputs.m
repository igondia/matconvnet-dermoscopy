function [y] = vl_nnmodulateInputs(x,wmod,varargin)
%VL_NNMODULATEINPUTS CNN dropout.
%   [Y] = VL_NNMODULATEINPUTS(x,wmod,varagin) modulate input x using the
%   weights in wmod
%
%   [DZDX] = VL_NNMODULATEINPUTS(x,wmod,dzdy) computes the
%   derivatives of the blocks projected onto DZDY. 


backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end
gpuMode = isa(x, 'gpuArray');

modChannels=size(wmod,3);

if ~backMode
    oSize=size(x);
    iChannels=size(x,3);
    outChannels=iChannels*modChannels;
    oSize(3)=outChannels;
    if(gpuMode)
        y=gpuArray(zeros(oSize,'single'));
    	mod=gpuArray(imresize(gather(wmod),[oSize(1) oSize(2)],'nearest'));
    else
        y=zeros(oSize,'single');
        mod=imresize(wmod,[oSize(1) oSize(2)],'nearest');
    end
    
    for c=1:iChannels
        y(:,:,(c-1)*modChannels+1:c*modChannels,:)=bsxfun(@times,mod,x(:,:,c,:));
    end
%BackMode: We have to select the corresponding layer    
else
    oSize=size(x);
    iChannels=size(x,3);
    if(gpuMode)
        y=gpuArray(zeros(oSize,'single'));
    	mod=gpuArray(imresize(gather(wmod),[oSize(1) oSize(2)],'nearest'));
    else
        y=zeros(oSize,'single');
        mod=imresize(wmod,[oSize(1) oSize(2)],'nearest');
    end
    for c=1:iChannels
        y(:,:,c,:)=sum(dzdy(:,:,(c-1)*modChannels+1:c*modChannels,:).*mod,3);
    end
end