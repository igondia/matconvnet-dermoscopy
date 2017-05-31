classdef tmovemex < matlab.unittest.TestCase

  properties
    numLabs
  end

  properties (TestParameter)
    device = {'cpu', 'VLDT_GPU'}
  end

  methods (TestClassSetup)
    function openParpool(test)
      test.numLabs = 4 ;
      if ~isempty(gcp('nocreate'))
        delete(gcp) ;
      end
      cl = parcluster('local');
      cl.NumWorkers = test.numLabs ;
      parpool(cl,test.numLabs) ;
    end
  end

  methods (TestClassTeardown)
    function closeParpool(test)
      if ~isempty(gcp('nocreate'))
        delete(gcp) ;
      end
    end
  end

  methods (TestMethodTeardown)
    function reset(test)
      if ~isempty(gcp('nocreate'))
        spmd
          vl_tmove('reset') ;
        end
      end
    end
  end

  methods (Test)
    function test_basic(test, device)
      haveGpu = strcmp(device,'VLDT_GPU') ;
      T = 10 ;
      finalValue = (test.numLabs * (test.numLabs + 1)) / 2 * ...
          test.numLabs^(T-1) ;
      format = {'single', [10 10],     'x0', 'cpu', ;
                'single', [10 15 3 2], 'x1', 'cpu'  ;
                'double', [3 4 5 6],   'x2', 'cpu' } ;
      if haveGpu
        format = vertcat(format, ...
                         {'single', [10 10],     'x0g', 'VLDT_GPU', ;
                          'single', [10 15 3 2], 'x1g', 'VLDT_GPU'  ;
                          'double', [3 4 5 6],   'x2g', 'VLDT_GPU' }) ;
      end
      spmd
        if haveGpu
          gpuDevice(1) ;
        end
        for i = 1:size(format,1)
          x{i} = tmovemex.makeArray(format(i,:)) + labindex ;
        end
        labBarrier() ;
        vl_tmove('init',format,labindex,numlabs) ;
        for t = 1:T
          for i = 1:size(format,1),
            vl_tmove('push',format{i,3},x{i}) ;
          end
          for i = 1:size(format,1),
            x{i} = vl_tmove('pull',format{i,3}) ;
          end
        end
        for i = 1:size(format,1)
          assert(x{i}(1) == finalValue) ;
        end
      end
    end

    function test_inplace(test, device)
      haveGpu = strcmp(device,'VLDT_GPU') ;
      if ~haveGpu, return ; end
      T = 10 ;
      finalValue = (test.numLabs * (test.numLabs + 1)) / 2 * ...
          test.numLabs^(T-1) ;
      format = {'single', [10 10],     'x0', 'VLDT_GPU', ;
                'single', [10 15 3 2], 'x1', 'VLDT_GPU'  ;
                'double', [3 4 5 6],   'x2', 'VLDT_GPU' } ;
      gpuDevice(1) ;
      spmd
        for i = 1:size(format,1)
          x{i} = tmovemex.makeArray(format(i,:)) + labindex ;
        end
        vl_tmove('init',format,labindex,numlabs) ;
        for t = 1:T
          for i = 1:size(format,1),
            vl_tmove('push',format{i,3},x{i},'inplace') ;
          end
          for i = 1:size(format,1),
            vl_tmove('pull',format{i,3},'inplace') ;
          end
        end
        for i = 1:size(format,1)
          assert(x{i}(1) == finalValue) ;
        end
      end
    end

  end

  methods (Static)
    function x = makeArray(format)
      x = zeros(format{1,2},format{1,1}) ;
      if numel(format) > 3 && strcmp(format{4},'VLDT_GPU')
        x = gpuArray(x) ;
      end
    end
  end

end
