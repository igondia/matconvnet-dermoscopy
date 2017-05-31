%% 
% Test function to compare nn_bnorm and its VLDT_GPU/CPU implementation with
% using VLFEAT
%%

VLDT_GPU = false;
VLDT_GPU = true ;

T = 1 ;
x = randn(64,64,32,32,'single') ;
g = randn(32,1,'single') ;
b = randn(32,1,'single') ;

if VLDT_GPU
  x = gpuArray(x) ;
  g = gpuArray(g) ;
  b = gpuArray(b) ;
end

a=vl_nnbnorm(x,g,b);
a_=vl_nnbnorm_old(x,g,b);

vl_testsim(a,a_)
