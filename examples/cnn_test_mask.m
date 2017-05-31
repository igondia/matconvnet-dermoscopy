function [info] = cnn_test(net, imdb, getBatch, varargin)
%CNN_TRAIN  An example implementation of SGD for training CNNs
%    CNN_TRAIN() is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option). Multi-GPU
%    support is relatively primitive but sufficient to obtain a
%    noticable speedup.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 1024 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.test = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
opts.numArch = 1 ;
opts.GrayScale = 0 ;
opts.useGpu =  false;
opts.numSet =  1;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = 'auc' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.plotStatistics = true;
opts.auc_data=struct('labels',[],'predictions',[],'auc',[],'batches',[]);
opts = vl_argparse(opts, varargin) ;


if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isempty(opts.test), opts.test = find(imdb.images.set==3) ; end
if isnan(opts.val), opts.val = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

% vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% setup error calculation function
hasError = true ;
if isstr(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
      hasError = false ;
    case 'multiclass'
      opts.errorFunction = @errorerrorFunction_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1err', 'top5err'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'binerr'} ; end
    case 'auc'
      opts.errorFunction = @error_auc ;
      if isempty(opts.errorLabels), opts.errorLabels = {'auc'} ; end  
    otherwise
      error('Unknown error function ''%s''.', opts.errorFunction) ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;

bestAUC=0;
if(~isempty(opts.val))
    [~,stats,outs] = process_epoch(opts, getBatch, 1, opts.val, 0, imdb, net) ;
else
    [~,stats,outs] = process_epoch(opts, getBatch, 1, opts.test, 0, imdb, net) ;
end
info.outs=outs;
info.stats=stats ;


% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

m = min(5, size(predictions,3)) ;

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ;

% -------------------------------------------------------------------------
function err = error_binary(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
predictions = squeeze(predictions(:,:,2,:));
labels_signed=labels;
labels_signed(labels==1)=-1;
labels_signed(labels==2)=1;
error = bsxfun(@times, predictions, labels_signed) < 0 ;
err = sum(error(:)) ;

% -------------------------------------------------------------------------
function err = error_auc(opts, labels, res)
% -------------------------------------------------------------------------
%Si solo hay una salida
if(size(res(end-1).x,3)==1)
    predictions=squeeze(gather(res(end-1).x));
    posClass=1;
%Si tenemos mas de una    
else
    predictions=squeeze(gather(res(end-1).x));
    predictions=predictions(1,:)';
    labels=labels(1,:)';
    posClass=1;
    %Antiguo softmax
    %predictions = vl_nnsoftmax(gather(res(end-1).x)) ;
    %predictions=squeeze(predictions(:,:,2,:));
    %posClass=2;
end

tlabels=[opts.auc_data.labels;labels];
tpredictions=[opts.auc_data.predictions;predictions];
try
[X,Y,T,auc] = perfcurve(tlabels,tpredictions,posClass);
catch
    auc=0;
end
err.labels = labels;
err.predictions = predictions;
err.auc = auc;


% -------------------------------------------------------------------------
function err = error_none(opts, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function  [net_cpu,stats,outs,prof] = process_epoch(opts, getBatch, epoch, subset, learningRate, imdb, net_cpu)
% -------------------------------------------------------------------------

% move the CNN to GPU (if needed)
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net = vl_simplenn_move(net_cpu, 'gpu') ;
  one = gpuArray(single(1)) ;
else
  net = net_cpu ;
  net_cpu = [] ;
  one = single(1) ;
end

% assume validation mode if the learning rate is zero
training = learningRate > 0 ;
if training
  mode = 'train' ;
  evalMode = 'normal' ;
else
  mode = 'val' ;
  evalMode = 'test' ;
end

% turn on the profiler (if needed)
if opts.profile
  if numGpus <= 1
    prof = profile('info') ;
    profile clear ;
    profile on ;
  else
    prof = mpiprofile('info') ;
    mpiprofile reset ;
    mpiprofile on ;
  end
end

res = [] ;
mmap = [] ;
stats = [] ;
start = tic ;

for t=1:opts.batchSize:numel(subset)
  fprintf('%s: epoch %02d: %3d/%3d: ', mode, epoch, ...
          fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
  numDone = 0 ;
  error = [] ;
  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    [im, pcoords, labels, instanceWeights] = getBatch(imdb, batch) ;

    if opts.prefetch
      if s==opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      getBatch(imdb, nextBatch) ;
    end

    if numGpus >= 1
      im = gpuArray(im) ;
    end

    % evaluate the CNN
    
    net.layers{end}.opts=[];
    %Binary Labels
    if(sum(labels<0))
     net.layers{end}.opts.loss='logistic';
    %Category labels 
    else
     net.layers{end}.opts.loss='softmaxlog';
    end
    %Weights for AUC
    net.layers{end}.class = reshape(labels,[1 1 size(labels,1) size(labels,2)]) ;
    net.layers{end}.opts.instanceWeights=reshape(instanceWeights,[1 1 size(instanceWeights,1) size(instanceWeights,2)]);
     
    if training, dzdy = one; else, dzdy = [] ; end
    res = vl_simplenn_mask(net, im, pcoords, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'mode', evalMode, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync, ...
                      'cudnn', opts.cudnn) ;

    % accumulate training errors
    auxerr = opts.errorFunction(opts, labels, res);
    %AUC
    if(isstruct(auxerr))
        opts.auc_data.labels=[opts.auc_data.labels;auxerr.labels];
        opts.auc_data.predictions=[opts.auc_data.predictions;auxerr.predictions];
        opts.auc_data.batches=[opts.auc_data.batches;batch];
        opts.auc_data.auc=[opts.auc_data.auc;auxerr.auc];
%         error(2)=0;
        error = sum([error, [sum(double(gather(res(end).x))) ;reshape(auxerr.auc(end),[],1) ; ]],2) ;
    else
        error = sum([error, [sum(double(gather(res(end).x))) ;reshape(auxerr,[],1) ; ]],2) ;
    end

    numDone = numDone + numel(batch) ;
  end % next sub-batch
  %Outs information
  outs=opts.auc_data;
  
  % collect and print learning statistics
  time = toc(start) ;
  
  stats = sum([stats,[0 ; error]],2); % works even when stats=[]
  
  stats(1) = time ;
  n = t + batchSize - 1 ; % number of images processed overall
  speed = n/time ;
  fprintf('%.1f Hz%s\n', speed) ;
    
  m = n / max(1,numlabs) ; % num images processed on this lab only
  if(~isempty(opts.auc_data.auc))
      stats(3) = error(2)*m;
  end
  fprintf(' obj:%.3g', stats(2)/m) ;
  for i=1:numel(opts.errorLabels)
    fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/m) ;
  end
  
  fprintf(' [%d/%d]', numDone, batchSize);
  fprintf('\n') ;

  % collect diagnostic statistics
  if training & opts.plotDiagnostics
    switchfigure(2) ; clf ;
    diag = [res.stats] ;
    barh(horzcat(diag.variation)) ;
    set(gca,'TickLabelInterpreter', 'none', ...
      'YTickLabel',horzcat(diag.label), ...
      'YDir', 'reverse', ...
      'XScale', 'log', ...
      'XLim', [1e-5 1]) ;
    drawnow ;
  end

end

% switch off the profiler
if opts.profile
  if numGpus <= 1
    prof = profile('info') ;
    profile off ;
  else
    prof = mpiprofile('info');
    mpiprofile off ;
  end
else
  prof = [] ;
end

% bring the network back to CPU
if numGpus >= 1
  net_cpu = vl_simplenn_move(net, 'cpu') ;
else
  net_cpu = net ;
end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
if nargin >= 6
  numGpus = numel(mmap.Data) ;
else
  numGpus = 1 ;
end

for l=numel(net.layers):-1:1
  for j=1:numel(res(l).dzdw)

    % accumualte gradients from multiple labs (GPUs) if needed
    if numGpus > 1
      tag = sprintf('l%d_%d',l,j) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numGpus, labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
    end
% 
%     if(strcmp(net.layers{l}.type, 'bnorm'))
%         disp('bnorm');
%     end
    if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
      % special case for learning bnorm moments
      thisLR = net.layers{l}.learningRate(j) ;
      net.layers{l}.weights{j} = ...
        (1-thisLR) * net.layers{l}.weights{j} + ...
        (thisLR/batchSize) * res(l).dzdw{j} ;
    else
      % standard gradient training
      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = lr * net.layers{l}.learningRate(j) ;
      net.layers{l}.momentum{j} = ...
        opts.momentum * net.layers{l}.momentum{j} ...
        - thisDecay * net.layers{l}.weights{j} ...
        - (1 / batchSize) * res(l).dzdw{j} ;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
        thisLR * net.layers{l}.momentum{j} ;
    end

    % if requested, collect some useful stats for debugging
    if opts.plotDiagnostics
      variation = [] ;
      label = '' ;
      switch net.layers{l}.type
        case {'conv','convt'}
          variation = thisLR * mean(abs(net.layers{l}.momentum{j}(:))) ;
          if j == 1 % fiters
            base = mean(abs(net.layers{l}.weights{j}(:))) ;
            label = 'filters' ;
          else % biases
            base = mean(abs(res(l+1).x(:))) ;
            label = 'biases' ;
          end
          variation = variation / base ;
          label = sprintf('%s_%s', net.layers{l}.name, label) ;
      end
      res(l).stats.variation(j) = variation ;
      res(l).stats.label{j} = label ;
    end
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
  end
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
  end
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchfigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end
