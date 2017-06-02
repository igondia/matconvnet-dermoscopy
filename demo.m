%A demo that shows how our diagnosis network works
function demo()

addpath('matlab');
addpath('examples/melanomas');
vl_setupnn();

%Parameters
%Input image and mask
imagePath='data/images/image1.jpg';
maskPath='data/images/mask1.png';
%Size of the block being processed: Important for memory consumption
bsize=48;
%Parameters of data augmentation
numAngles=12;
numCrops=4;
%Number of GPUS
useGPU=true;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              Step 1: Load the nets           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Load the segmentation network
load('models/segmentationNet.mat');
isDag = isfield(net, 'params') ;
if(isDag)
    opts.networkType = 'dagnn' ;
    net_seg = dagnn.DagNN.loadobj(net) ;
    %Remove the weak loss and the performance
    net_seg.removeLayer('weakloss'); 
    net_seg.removeLayer('performance'); 
    %Add a layer to compute a probability output map
    net_seg.addLayer('wprob' , ...
             dagnn.SoftMax('gamma',1.0),...
             'prediction',...
             'wprob');         
else
    net_seg=vl_simplenn_clean(net);
    %Substitute the last layer (weak loss) by a softmax
    net_seg.layers{end}=struct('type', 'softmax', 'gamma', 1);
    clear net;
end
net_seg.mode='test';

%Now load diagnosis network
net=load('models/diagnosisNet.mat');
if isfield(net, 'net') ;
  net = net.net ;
end
opts.networkType = 'dagnn' ;
net = dagnn.DagNN.loadobj(net) ;
net.mode='test';
%Move nets to GPU is case is available
if useGPU
  net.move('gpu');
  if(isDag)
      net_seg.move('gpu');
  else
    net_seg = vl_simplenn_move(net_seg, 'gpu') ;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Step 2: Load and preprocess and image        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im=imread(imagePath);
%figure(1);imshow(im);
mask=imread(maskPath)>128;
imSize=net.meta.inputs.size(1:2);
%Data Augmentation
[im,pcoords]=dataAugmentation(im,mask,imSize,numAngles,numCrops);
%Convert image to float and normalize it
im=single(im);
im=bsxfun(@minus,im,reshape(net.meta.normalization.vdata_mean,[1 1 3]));
numVariations=size(im,4);
%Convert to GPU in case is necessary
if useGPU
    im=gpuArray(im);
    pcoords=gpuArray(pcoords);
end
%Loop of blocks to avoid memory overload in CPU or GPU
numBlocks=ceil(numVariations/bsize);
numClases=length(net.meta.classes.name);
output=zeros(numClases,numVariations);
for b=1:numBlocks
    fprintf('*************\n');
    fprintf('**Block %d/%d**\n',b,numBlocks);
    fprintf('*************\n');
    idxIm=(b-1)*bsize+1:min(b*bsize,numVariations);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Step 3: Compute the segmentation map of dermoscopic features  %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ç
    fprintf('Segmenting images into dermoscopic features\n');
    sinputs={'input',im(:,:,:,idxIm)};
    net_seg.eval(sinputs);   
    wmod=getModulatingWeights(net_seg,pcoords(:,:,1,idxIm)>0);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%         Step 4: Setup the diagnosis net                      %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Remove loss and performance and add a softmax
    net=setupNet(net,wmod,pcoords(:,:,:,idxIm));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%           Step 5: Run the diagnosis net                      %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('Computing the diagnosis\n');
    net.eval(sinputs);
    output(:,idxIm)=squeeze(gather(net.vars(end).value));
end
%Fuse the outputs for the variations with a simple average
output=mean(output,2);
%Get probs using a softmax
output=softmax(output);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                    Step 6: Show Results                      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Output probs:\n');
for c=1:length(net.meta.classes.name)
    fprintf('\t%f: %s\n',output(c),net.meta.classes.name{c});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Auxiliary function that gets the modulation weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function wmod = getModulatingWeights(net_seg,mask)
isgpu=isa(mask,'gpuArray');
%Get the modulating weights
wmod=gather(net_seg.vars(end).value(:,:,2:end,:));
maskr=imresize(gather(mask),[size(wmod,1) size(wmod,2)],'Method','nearest');
if(isgpu)
    maskr=gpuArray(maskr);
end
wmod=bsxfun(@times,wmod,maskr);
%Concat the original features by introducin a map of ones
wmod = cat(3,ones(size(wmod,1),size(wmod,2),1,size(wmod,4),'single'),wmod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Auxiliary function that setups the network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function net=setupNet(net,wmod,pcoords)

selLossPerf = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block}));
for l=1:length(selLossPerf)
    names{l}=net.layers(selLossPerf(l)).name;
end
for l=1:length(selLossPerf)
    %Get the previous layer
    net.removeLayer(names{l});
end

%Add the param pcoords in case it is necessary 
selPolarPool = find(cellfun(@(x) isa(x,'dagnn.CircPoolingMask'), {net.layers.block}));
if(~isempty(selPolarPool))
    p = net.layers(selPolarPool).paramIndexes(1);
    net.params(p).value=pcoords;
end

%Add wmods to the modulation block
selModulation = find(cellfun(@(x) isa(x,'dagnn.Modulation'), {net.layers.block}));
if (~isempty(selModulation))
    p = net.layers(selModulation).paramIndexes(1);
    net.params(p).value=wmod;
end
