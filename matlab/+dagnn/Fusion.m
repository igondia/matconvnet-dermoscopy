classdef Fusion < dagnn.ElementWise
  
  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnfuseInputs(inputs{1}, inputs{2}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnfuseInputs(inputs{1}, inputs{2}, derOutputs{1}) ;
      derParams = {} ;
    end

    
    function obj = Fusion(varargin)
      obj.load(varargin) ;
    end
  end
end
