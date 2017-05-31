classdef Aggregation < dagnn.Filter
    properties
        method = 'max'
        r = 1.0;
        opts = {}
    end
  methods
    function outputs = forward(obj, inputs, params)
      gpuMode = isa(inputs{1}, 'gpuArray') ; 
      validos=single(inputs{2}(:,:,1,:)>0);
      validos = imresize(validos,[size(inputs{1},1) size(inputs{1},2)],'Method','nearest');
      if(gpuMode>0)
          validos = gpuArray(validos) ;
      end
      outputs{1} = vl_nnaggregation(inputs{1}, validos, 'r', obj.r,'method', obj.method,obj.opts{:}) ;
      if(gpuMode>0)
          clear validos;
      end
 
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
     
        gpuMode = isa(inputs{1}, 'gpuArray') ; 
        validos=single(inputs{2}(:,:,1,:)>0);
        validos = imresize(validos,[size(inputs{1},1) size(inputs{1},2)],'Method','nearest');
        if(gpuMode>0)
            validos = gpuArray(validos) ;
        end
        derInputs{1} = vl_nnaggregation(inputs{1}, validos, derOutputs{1}, 'r', obj.r,'method', obj.method,obj.opts{:}) ;       
        if(gpuMode>0)
            clear validos;
        end

      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = zeros(1,4);
      outputSizes{1}(1) = 1;
      outputSizes{1}(2) = 1;
      outputSizes{1}(3) = inputSizes{1}(3) ;
      outputSizes{1}(4) = inputSizes{1}(4) ;
    end

    function obj = Aggregation(varargin)
      obj.load(varargin) ;
    end
    
  end
end
