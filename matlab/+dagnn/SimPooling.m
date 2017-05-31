classdef SimPooling < dagnn.Filter
  properties
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(self, inputs, params)
        
        outputs{1} = vl_nnsimpool(inputs{1}, self.opts{:}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
     
        derInputs{1} = vl_nnsimpool(inputs{1}, derOutputs{1},self.opts{:}) ;
        
      derParams = {} ;
    end

    function obj = SimPooling(varargin)
      obj.load(varargin) ;
    end
  end
end
