classdef WeakLoss < dagnn.Loss
  properties
      lambda=[];
      A=[];
      b=[];
      beta=[];
      maxLambda=10;
      gamma=20;
  end
  methods
    function outputs = forward(obj, inputs, params)
      %inputs 1 x, 2 pcoords, 3 labels
      
      %outputs 1 x 2 new_lambda
      gpuMode = isa(inputs{1}, 'gpuArray') ; 
      pcoordsr = params{1};
      pcoordsr = gpuArray(imresize(gather(params{1}),[size(inputs{1},1) size(inputs{1},2)],'Method','nearest'));
      pcoordsr(:,:,1,:)=bsxfun(@rdivide,pcoordsr(:,:,1,:),max(max(pcoordsr(:,:,1,:),[],1),[],2));      
      
      obj.lambda=zeros([2*size(inputs{2},1) size(inputs{2},2)],'single');
      %Return the outputs and the new lambda
      [outputs{1}, new_lambda]= vl_nnweakloss(inputs{1}.*obj.gamma, pcoordsr, gpuArray(single(inputs{2})), gpuArray(obj.lambda),gpuArray(obj.A),gpuArray(obj.b),gpuArray(obj.beta),'maxLambda',obj.maxLambda) ;
      obj.lambda=new_lambda;
%       obj.lambda
      if(gpuMode>0)
          clear pcoordsr;
      end
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        %inputs 1 x, 2 pcoords, 3 labels
        %outputs 1 x 2 new_lambda
      
        gpuMode = isa(inputs{1}, 'gpuArray') ; 
        pcoordsr = params{1};
        pcoordsr = gpuArray(imresize(gather(params{1}),[size(inputs{1},1) size(inputs{1},2)],'Method','nearest'));
        pcoordsr(:,:,1,:)=bsxfun(@rdivide,pcoordsr(:,:,1,:),max(max(pcoordsr(:,:,1,:),[],1),[],2));      
        [derInputs{1}, aux]= vl_nnweakloss(inputs{1}.*obj.gamma, pcoordsr, gpuArray(single(inputs{2})), gpuArray(obj.lambda), gpuArray(obj.A),gpuArray(obj.b),gpuArray(obj.beta),gpuArray(single(derOutputs{1})),'maxLambda',obj.maxLambda);         
        derInputs{1}=derInputs{1}*obj.gamma;
        derInputs{2} = [] ;
        derParams = {} ;
    end

    function obj = WeakLoss(varargin)
      obj.load(varargin) ;
    end
    
    function params = initParams(obj)
      % We set it to zero
      params{1} = 0 ;
      
    end

  end
end
