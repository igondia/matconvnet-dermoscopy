classdef CircPoolingMask < dagnn.Filter
  properties
    method = 'max'
    poolSize = [5 5]
    overlap = [0.0 0.0]
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(self, inputs, params)
        gpuMode = isa(inputs{1}, 'gpuArray') ;
            
        %Ojo pcoords (input{2}) ya tiene que venir en GPU
        pcoordsr=params{1};
        pcoordsr(:,:,1,:)=bsxfun(@rdivide,pcoordsr(:,:,1,:),max(max(pcoordsr(:,:,1,:),[],1),[],2));
        pcoordsr(pcoordsr<0)=1;
        pcoordsr = imresize(gather(pcoordsr),[size(inputs{1},1) size(inputs{1},2)]);
        if(gpuMode)
            pcoordsr=gpuArray(pcoordsr);
        end
        pcoordsr(:,:,1,:)=bsxfun(@rdivide,pcoordsr(:,:,1,:),max(max(pcoordsr(:,:,1,:),[],1),[],2));
        outputs{1} = vl_nncircpool_mask(inputs{1}, pcoordsr, self.poolSize, 'pad', self.pad, 'overlap', self.overlap, 'method', self.method, self.opts{:}) ;
       
        clear pcoordsr;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      gpuMode = isa(inputs{1}, 'gpuArray') ;
            
        %Ojo pcoords (input{2}) ya tiene que venir en GPU
        pcoordsr=params{1};
        pcoordsr(:,:,1,:)=bsxfun(@rdivide,pcoordsr(:,:,1,:),max(max(pcoordsr(:,:,1,:),[],1),[],2));
        pcoordsr(pcoordsr<0)=1;
        pcoordsr = gpuArray(imresize(gather(pcoordsr),[size(inputs{1},1) size(inputs{1},2)]));
        pcoordsr(:,:,1,:)=bsxfun(@rdivide,pcoordsr(:,:,1,:),max(max(pcoordsr(:,:,1,:),[],1),[],2));
        
        if(gpuMode)
            pcoordsr=gpuArray(pcoordsr);
        end
        derInputs{1} = vl_nncircpool_mask(inputs{1}, pcoordsr, self.poolSize, derOutputs{1}, 'pad', self.pad, 'overlap', self.overlap, 'method', self.method, self.opts{:}) ;       
        clear pcoordsr;
        
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = zeros(1,4);
      outputSizes{1}(1) = obj.poolSize(1);
      outputSizes{1}(2) = obj.poolSize(2);
      outputSizes{1}(3) = inputSizes{1}(3) ;
      outputSizes{1}(4) = inputSizes{1}(4) ;
    end

    function obj = CircPoolingMask(varargin)
      obj.load(varargin) ;
    end
    
    function params = initParams(obj)
      % We set it to zero
      
      params{1} = 0 ;
      
    end

  end
end
