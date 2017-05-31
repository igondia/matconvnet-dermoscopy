classdef Modulation < dagnn.ElementWise
  
  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnmodulateInputs(inputs{1}, params{1}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnmodulateInputs(inputs{1}, params{1}, derOutputs{1}) ;
      derParams = {} ;
    end

    function params = initParams(obj)
      % We set it to zero
      
      params{1} = 0 ;
      
    end

    
    function obj = Modulation(varargin)
      obj.load(varargin) ;
    end
  end
end
