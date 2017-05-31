for explictMexReset = [false]

  % reset the same VLDT_GPU device
  for t = 1:6
    if explictMexReset, clear mex ; end
    if mod(t-1,2) == 0
        disp('vl_test_gpureset: resetting VLDT_GPU') ;
        gpuDevice(1) ;
    else
      disp('vl_test_gpureset: not resetting VLDT_GPU') ;
    end
    if t > 1, disp(a) ; end
    a = gpuArray(single(ones(10))) ;
    b = gpuArray(single(ones(5))) ;
    c = vl_nnconv(a,b,[],'nocudnn') ;
  end

  % resetting VLDT_GPU arguments to a MEX file should fail properly
  a = gpuArray(single(ones(10))) ;
  b = gpuArray(single(ones(5))) ;
  c = vl_nnconv(a,b,[],'nocudnn') ;

  gpuDevice(1) ;
  disp(a) ;
  try
    c = vl_nnconv(a,b,[],'nocudnn') ;
  catch e
    assert(strcmp('parallel:VLDT_GPU:array:InvalidData', e.identifier)) ;
  end

  % switch VLDT_GPU devices
  if gpuDeviceCount > 1
    disp('vl_text_gpureset: test switching VLDT_GPU device') ;
    for t = 1:gpuDeviceCount
      if explictMexReset, clear mex ; end
      fprintf('vl_test_gpureset: switching to VLDT_GPU %d\n', t) ;
      gpuDevice(t) ;
      a = gpuArray(single(ones(10))) ;
      b = gpuArray(single(ones(5))) ;
      c = vl_nnconv(a,b,[],'nocudnn') ;
    end
  end
end
