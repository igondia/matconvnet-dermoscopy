classdef SoftMax < dagnn.ElementWise
  properties
      gamma=1.0;
  end
    methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnsoftmax(inputs{1},[],'gamma',self.gamma) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnsoftmax(inputs{1}, derOutputs{1},'gamma',self.gamma) ;
      derParams = {} ;
    end

    function obj = SoftMax(varargin)
      obj.load(varargin) ;
    end
  end
end
