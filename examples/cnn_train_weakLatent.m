function [net, stats] = cnn_train(net, imdb, getBatch, varargin)
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
%    (specify the list of VLDT_GPU IDs in the `gpus` option).

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.saveMomentum = true ;
opts.nesterovUpdate = false ;
opts.useGpu =  false;
opts.GrayScale = 0 ;
opts.randomSeed = 0 ;
opts.numArch = 1 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = 'weak_seg' ;
opts.perf_data=struct('labels',[],'predictions',[],'perf',[]);
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.plotStatistics = true;
opts.validLabelsError = 1;

opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
% if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
% if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

net = vl_simplenn_tidy(net); % fill in some eventually missing values
net.layers{end-1}.precious = 1; % do not remove predictions, used for error
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  for i=1:numel(net.layers)
    J = numel(net.layers{i}.weights) ;
    if ~isfield(net.layers{i}, 'learningRate')
      net.layers{i}.learningRate = ones(1, J) ;
    end
    if ~isfield(net.layers{i}, 'weightDecay')
      net.layers{i}.weightDecay = ones(1, J) ;
    end
  end
end

% setup error calculation function
hasError = true ;
if isstr(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
      hasError = false ;
    case 'multiclass'
      opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1err', 'top5err'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'binerr'} ; end
    case 'auc'
      opts.errorFunction = @error_auc ;
      if isempty(opts.errorLabels), opts.errorLabels = {'auc'} ; end  
     case 'ap'
      opts.errorFunction = @error_ap ;
      if isempty(opts.errorLabels), opts.errorLabels = {'ap'} ; end    
     case 'weak_seg'
      opts.errorFunction = @error_wseg ;
      opts.b=abs(net.layers{end}.b);
      if isempty(opts.errorLabels), opts.errorLabels = {'wseg'} ; end     
     case 'weak_seg_bal'
      opts.errorFunction = @error_wseg_bal ;
      opts.b=abs(net.layers{end}.b);
      if isempty(opts.errorLabels), opts.errorLabels = {'wsegbal'} ; end      
    otherwise
      error('Unknown error function ''%s''.', opts.errorFunction) ;
  end
end

state.getBatch = getBatch ;
stats = [] ;

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net, state, stats] = loadState(modelPath(start)) ;
else
  state = [] ;
end

for epoch=start+1:opts.numEpochs

  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.

  rng(epoch + opts.randomSeed) ;
  prepareGPUs(opts, epoch == start+1) ;

  % Train for one epoch.
  params = opts ;
  params.epoch = epoch ;
  params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  params.val = opts.val(randperm(numel(opts.val))) ;
  params.imdb = imdb ;
  params.getBatch = getBatch ;

  if numel(opts.gpus) <= 1
    [net, state] = processEpoch(net, state, params, 'train') ;
    [net, state] = processEpoch(net, state, params, 'val') ;
    if ~evaluateMode
      saveState(modelPath(epoch), net, state) ;
    end
    lastStats = state.stats ;
  else
    spmd
      [net, state] = processEpoch(net, state, params, 'train') ;
      [net, state] = processEpoch(net, state, params, 'val') ;
      if labindex == 1 && ~evaluateMode
        saveState(modelPath(epoch), net, state) ;
      end
      lastStats = state.stats ;
    end
    lastStats = accumulateStats(lastStats) ;
  end

  stats.train(epoch) = lastStats.train ;
  stats.val(epoch) = lastStats.val ;
  clear lastStats ;
  saveStats(modelPath(epoch), stats) ;

  if params.plotStatistics
    switchFigure(1) ; clf ;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats.train)', ...
      fieldnames(stats.val)'), {'num', 'time','esegbal'},'stable') ;
    for p = plots
      p = char(p) ;
      values = zeros(0, epoch) ;
      leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
          tmp = [stats.(f).(p)] ;
          values(end+1,:) = tmp(1,:)' ;
          leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ;
      plot(1:epoch, values','o-') ;
      xlabel('epoch') ;
      title(p) ;
      legend(leg{:},'Location','northwest') ;
      grid on ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
  end
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end

% -------------------------------------------------------------------------
function err = error_multiclass(params, labels, res)
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
function err = error_binary(params, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
error = bsxfun(@times, predictions, labels) < 0 ;
err = sum(error(:)) ;

% -------------------------------------------------------------------------
function err = error_auc(params, labels, res,pcoords)
% -------------------------------------------------------------------------
%Si solo hay una salida

if(size(res(end-1).x,3)==1)
    predictions=squeeze(gather(res(end-1).x));
    posClass=1;
    labels=labels';
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

tlabels=[params.perf_data.labels;labels];
tpredictions=[params.perf_data.predictions;predictions];
[X,Y,T,auc] = perfcurve(labels,predictions,posClass);
err.labels = tlabels;
err.predictions = tpredictions;
err.perf = auc;

% -------------------------------------------------------------------------
function err = error_ap(params, labels, res, pcoords)
% -------------------------------------------------------------------------
%Si solo hay una salida

if(size(res(end-1).x,1)>1 || size(res(end-1).x,2)>1)
    predictions=squeeze(gather(res(end-1).x));
    predictions=vl_nnsoftmax(predictions);
    pcoordsr = imresize(pcoords,[size(predictions,1) size(predictions,2)],'Method','nearest'); 
    validos=pcoordsr(:,:,1,:)>=0;
    predictions=predictions.*repmat(validos,[1 1 size(predictions,3) 1]);
    predictions=squeeze(mean(mean(predictions,1),2))';    
%     predictions=squeeze(max(max(predictions,[],1),[],2))';    
else
    predictions=squeeze(gather(res(end-1).x(:,:,params.validLabelsError>0,:)))';
end
labels=labels(params.validLabelsError>0,:)';
labels(labels<0)=0;
predictions=predictions(:,2:end);
labels=labels(:,2:end);
tlabels=[params.perf_data.labels;labels];
tpredictions=[params.perf_data.predictions;predictions];
numPat=size(tlabels,2);
ap=zeros(1,numPat+1);
%El global
% ap(1) = computeAP(predictions(:),labels(:)>0);
%El de los patrones
for i=1:1:numPat
    ap(i+1) = computeAP(tpredictions(:,i),tlabels(:,i)>0); 
    if(isnan(ap(i+1)))
        disp('error ap nan');
        ap(i+1)=0;
    end
end
%El promedio
%ap(1) = mean(ap(2:end));
ap(1) = computeAP(predictions(:),labels(:)>0);
err.labels = labels;
err.predictions = predictions;
err.perf = ap;

% -------------------------------------------------------------------------
function err = error_wseg(params, labels, res, pcoords)
% -------------------------------------------------------------------------
%Si solo hay una salida

if(size(res(end-1).x,1)>1 || size(res(end-1).x,2)>1)
    predictions=squeeze(gather(res(end-1).x));
    predictions=vl_nnsoftmax(predictions);
    pcoordsr = imresize(pcoords,[size(predictions,1) size(predictions,2)],'Method','nearest'); 
    validos=pcoordsr(:,:,1,:)>=0;
    predictions=predictions.*repmat(validos,[1 1 size(predictions,3) 1]);
    predictions=squeeze(sum(sum(predictions,1),2))';    
    predictions=bsxfun(@rdivide,predictions,sum(predictions,2));
%     predictions=squeeze(max(max(predictions,[],1),[],2))';    
else
    predictions=squeeze(gather(res(end-1).x(:,:,params.validLabelsError>0,:)))';
end
labels=labels(params.validLabelsError>0,:)';
labels(labels<0)=0;
%Now we can decide the final predictions based on the A,b
label_pred=zeros(size(labels));
label_pred(predictions<=params.b(1))=0;
label_pred(predictions>=params.b(3) & predictions<=params.b(4))=1;
label_pred(predictions>=params.b(5))=2;

tlabels=[params.perf_data.labels;labels];
tpredictions=[params.perf_data.predictions;label_pred];
numPat=size(tlabels,2);
acc=zeros(1,numPat+1);
%El global
acc(1) = sum(tpredictions(:)==tlabels(:))/length(tlabels(:));
%El de los patrones
for i=1:1:numPat
    acc(i+1) = sum(tpredictions(:,i)==tlabels(:,i))/length(tlabels(:,i)); 
end

err.labels = labels;
err.predictions = label_pred;
err.perf = acc;


% -------------------------------------------------------------------------
function err = error_wseg_bal(params, labels, res, pcoords)
% -------------------------------------------------------------------------
%Si solo hay una salida
numCat=size(res(end-1).x,3);
numIm=size(res(end-1).x,4);
if(size(res(end-1).x,1)>1 || size(res(end-1).x,2)>1)
    predictions=squeeze(gather(res(end-1).x));
    %Eliminamos el background
%     predictions(:,:,1,:)=-1000;
    predictions=vl_nnsoftmax(predictions,[],'gamma',25);
    pcoordsr = imresize(pcoords,[size(predictions,1) size(predictions,2)],'Method','nearest'); 
    validos=pcoordsr(:,:,1,:)>=0;
    predictions=predictions.*repmat(validos,[1 1 size(predictions,3) 1]);
    vals=max(predictions,[],3);
    %We remove non-maximal values
    predictions(bsxfun(@lt,predictions,vals))=0; 
    predictions=squeeze(sum(sum(predictions,1),2))';    
    predictions=bsxfun(@rdivide,predictions,sum(predictions,2));
%     predictions=squeeze(max(max(predictions,[],1),[],2))';    
else
    predictions=squeeze(gather(res(end-1).x(:,:,params.validLabelsError>0,:)))';
end
labels=labels(params.validLabelsError>0,:)';
labels(labels<0)=0;
%Now we can decide the final predictions based on the A,b
label_pred=ones(size(labels));
minVal=0.05;%mean(params.b([1 3]));
label_pred(predictions<=minVal)=0;
% label_pred(predictions>=params.b(3) & predictions<=params.b(4))=1;
% label_pred(predictions>=params.b(5))=2;
[sum(label_pred);sum(labels>0)]

%Remove the BG from the analysis
labels=labels(:,2:end);
predictions=predictions(:,2:end);
%Check how many conditions we fulfill
tlabels=[params.perf_data.labels;labels];
tpredictions=[params.perf_data.predictions;predictions];
numPat=size(tlabels,2);
acc=zeros(1,numPat+1);
%El global
nn=sum(tlabels(:)==0);
np1=sum(tlabels(:)==1 | tlabels(:)==3);
np2=sum(tlabels(:)==2);
ps=[];
if(nn>0)
    pn=sum(tpredictions(tlabels(:)==0)<=0.05)/nn;
    ps=[ps pn];
end
if(np1>0)
    pp1=sum(tpredictions(tlabels(:)==1)>=0.05 & tpredictions(tlabels(:)==1)<=0.6 );
    pp1=pp1+sum(tpredictions(tlabels(:)==3)>=0.05 & tpredictions(tlabels(:)==3)<=0.5 );
    pp1=pp1/np1;
    ps=[ps pp1];
end
if(np2>0)
    pp2=sum(tpredictions(tlabels(:)==2)>=0.5)/np2;
    ps=[ps pp2];
end
[ps mean(ps)]
% acc(i+1) =mean(ps);
%El de los patrones
for i=1:1:numPat
    nn=sum(tlabels(:,i)==0);
    np1=sum(tlabels(:,i)==1 | tlabels(:,i)==3);
    np2=sum(tlabels(:,i)==2);
    ps=[];
    if(nn>0)
        pn=sum(tpredictions(tlabels(:,i)==0,i)<=0.05)/nn;
        ps=[ps pn];
    end
    if(np1>0)
        
        pp1=sum(tpredictions(tlabels(:,i)==1,i)>=0.05 & tpredictions(tlabels(:,i)==1,i)<=0.6 );
        pp1=pp1+sum(tpredictions(tlabels(:,i)==3,i)>=0.05 & tpredictions(tlabels(:,i)==3,i)<=0.5 );
        pp1=pp1/np1;
        ps=[ps pp1];
    end
    if(np2>0)
        pp2=sum(tpredictions(tlabels(:,i)==2,i)>=0.5)/np2;
        ps=[ps pp2];
    end
    
    acc(i+1) =mean(ps); 
end
acc(1) = mean(acc(2:end));
err.labels = labels;
err.predictions = predictions;
err.perf = acc*numIm;

% -------------------------------------------------------------------------
function err = error_none(params, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.
% initialize with momentum 0
if isempty(state) || isempty(state.momentum)
  for i = 1:numel(net.layers)
    for j = 1:numel(net.layers{i}.weights)
      state.momentum{i}{j} = 0 ;
    end
  end
end

% move CNN  to VLDT_GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
  net = vl_simplenn_move(net, 'gpu') ;
  for i = 1:numel(state.momentum)
    for j = 1:numel(state.momentum{i})
      state.momentum{i}{j} = gpuArray(state.momentum{i}{j}) ;
    end
  end
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end
if numGpus > 1
  parserv = ParameterServer(params.parameterServer) ;
  vl_simplenn_start_parserv(net, parserv) ;
else
  parserv = [] ;
end

% profile
if params.profile
  if numGpus <= 1
    profile clear ;
    profile on ;
  else
    mpiprofile reset ;
    mpiprofile on ;
  end
end

subset = params.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;
res = [] ;
error = [] ;

start = tic ;
for t=1:params.batchSize:numel(subset)
  fprintf('%s: epoch %02d: %3d/%3d:', mode, params.epoch, ...
          fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
  batchSize = min(params.batchSize, numel(subset) - t + 1) ;

  for s=1:params.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+params.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    [im, pcoords, labels, instanceWeights] = params.getBatch(params.imdb, batch) ;
    

    if params.prefetch
      if s == params.numSubBatches
        batchStart = t + (labindex-1) + params.batchSize ;
        batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
      params.getBatch(params.imdb, nextBatch) ;
    end

    if numGpus >= 1
      im = gpuArray(im) ;
    end

   %Convert labels to single
    net.layers{end}.labels = single(labels) ;
    
    if(~isfield(net.layers{end}, 'lambda') || size(net.layers{end}.lambda,2)~=size(labels,2))
        net.layers{end}.lambda=zeros([2*size(labels,1) size(labels,2)],'single');
    end
    if numGpus >= 1
      net.layers{end}.lambda=gpuArray(net.layers{end}.lambda);
      net.layers{end}.labels=gpuArray(net.layers{end}.labels);
    end
    
    
    if strcmp(mode, 'train')
      dzdy = one ;
      evalMode = 'normal' ;
    else
      dzdy = [] ;
      evalMode = 'test' ;
    end
    [res net]= vl_simplenn_mask(net, im, pcoords, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'mode', evalMode, ...
                      'conserveMemory', params.conserveMemory, ...
                      'backPropDepth', params.backPropDepth, ...
                      'sync', params.sync, ...
                      'cudnn', params.cudnn, ...
                      'parameterServer', parserv, ...
                      'holdOn', s < params.numSubBatches) ;

     % accumulate training errors
    auxerr = params.errorFunction(params, labels, res,pcoords);
    %AUC or AP
    if(isstruct(auxerr))
        params.perf_data.labels=[params.perf_data.labels;auxerr.labels];
        params.perf_data.predictions=[params.perf_data.predictions;auxerr.predictions];
        params.perf_data.perf=[params.perf_data.perf;auxerr.perf];
        mean_perf=params.perf_data.perf(end,:);
%         error(2)=0;
        error = sum([error, [sum(double(gather(res(end).x))) ;reshape(mean_perf(1),[],1) ; reshape(mean_perf(2:end),[],1) ]],2) ;
    else
        error = sum([error, [sum(double(gather(res(end).x))) ;reshape(auxerr,[],1) ; ]],2) ;
    end
  end

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(parserv), parserv.sync() ; end
    [net, res, state] = accumulateGradients(net, res, state, params, batchSize, parserv) ;
  end

  % get statistics
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats = extractStats(net, params, error / num) ;
  stats.num = num ;
  stats.time = time ;
  currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  if t == 3*params.batchSize + 1
    % compensate for the first three iterations, which are outliers
    adjustTime = 4*batchTime - time ;
    stats.time = time + adjustTime ;
  end

  fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'},'stable')
    f = char(f) ;
    fprintf(' %s: ', f) ;
    for j=1:length(stats.(f))
        fprintf('%.3f ', stats.(f)(j)) ;
    end
  end
  fprintf('\n') ;

  % collect diagnostic statistics
  if strcmp(mode, 'train') && params.plotDiagnostics
    switchFigure(2) ; clf ;
    diagn = [res.stats] ;
    diagnvar = horzcat(diagn.variation) ;
    diagnpow = horzcat(diagn.power) ;
    subplot(2,2,1) ; barh(diagnvar) ;
    set(gca,'TickLabelInterpreter', 'none', ...
      'YTick', 1:numel(diagnvar), ...
      'YTickLabel',horzcat(diagn.label), ...
      'YDir', 'reverse', ...
      'XScale', 'log', ...
      'XLim', [1e-5 1], ...
      'XTick', 10.^(-5:1)) ;
    grid on ;
    subplot(2,2,2) ; barh(sqrt(diagnpow)) ;
    set(gca,'TickLabelInterpreter', 'none', ...
      'YTick', 1:numel(diagnpow), ...
      'YTickLabel',{diagn.powerLabel}, ...
      'YDir', 'reverse', ...
      'XScale', 'log', ...
      'XLim', [1e-5 1e5], ...
      'XTick', 10.^(-5:5)) ;
    grid on ;
    subplot(2,2,3); plot(squeeze(res(end-1).x)) ;
    drawnow ;
  end
end

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
  if numGpus <= 1
    state.prof.(mode) = profile('info') ;
    profile off ;
  else
    state.prof.(mode) = mpiprofile('info');
    mpiprofile off ;
  end
end
if ~params.saveMomentum
  state.momentum = [] ;
else
  for i = 1:numel(state.momentum)
    for j = 1:numel(state.momentum{i})
      state.momentum{i}{j} = gather(state.momentum{i}{j}) ;
    end
  end
end

net = vl_simplenn_move(net, 'cpu') ;

% -------------------------------------------------------------------------
function [net, res, state] = accumulateGradients(net, res, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for l=numel(net.layers):-1:1
  for j=numel(res(l).dzdw):-1:1

    if ~isempty(parserv)
      tag = sprintf('l%d_%d',l,j) ;
      parDer = parserv.pull(tag) ;
    else
      parDer = res(l).dzdw{j}  ;
    end

    if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
      % special case for learning bnorm moments
      thisLR = net.layers{l}.learningRate(j) ;
      net.layers{l}.weights{j} = vl_taccum(...
        1 - thisLR, ...
        net.layers{l}.weights{j}, ...
        thisLR / batchSize, ...
        parDer) ;
    else
      % Standard gradient training.
      thisDecay = params.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = params.learningRate * net.layers{l}.learningRate(j) ;

      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, net.layers{l}.weights{j}) ;

        % Update momentum.
        state.momentum{l}{j} = vl_taccum(...
          params.momentum, state.momentum{l}{j}, ...
          -1, parDer) ;

        % Nesterov update (aka one step ahead).
        if params.nesterovUpdate
          delta = vl_taccum(...
            params.momentum, state.momentum{l}{j}, ...
            -1, parDer) ;
        else
          delta = state.momentum{l}{j} ;
        end

        % Update parameters.
        net.layers{l}.weights{j} = vl_taccum(...
          1, net.layers{l}.weights{j}, ...
          thisLR, delta) ;
      end
    end

    % if requested, collect some useful stats for debugging
    if params.plotDiagnostics
      variation = [] ;
      label = '' ;
      switch net.layers{l}.type
        case {'conv','convt'}
          variation = thisLR * mean(abs(state.momentum{l}{j}(:))) ;
          power = mean(res(l+1).x(:).^2) ;
          if j == 1 % fiters
            base = mean(net.layers{l}.weights{j}(:).^2) ;
            label = 'filters' ;
          else % biases
            base = sqrt(power) ;%mean(abs(res(l+1).x(:))) ;
            label = 'biases' ;
          end
          variation = variation / base ;
          label = sprintf('%s_%s', net.layers{l}.name, label) ;
      end
      res(l).stats.variation(j) = variation ;
      res(l).stats.power = power ;
      res(l).stats.powerLabel = net.layers{l}.name ;
      res(l).stats.label{j} = label ;
    end
  end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  % initialize stats stucture with same fields and same order as
  % stats_{1}
  stats__ = stats_{1} ;
  names = fieldnames(stats__.(s))' ;
  values = zeros(1, numel(names)) ;
  fields = cat(1, names, num2cell(values)) ;
  stats.(s) = struct(fields{:}) ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net, params, errors)
% -------------------------------------------------------------------------
stats.objective = errors(1) ;
for i = 1:numel(params.errorLabels)
  stats.(params.errorLabels{i}) = errors(i+1) ;
end
if(numel(errors)-1>numel(params.errorLabels))
    stats.esegbal=errors(numel(params.errorLabels)+2:end);
end
% -------------------------------------------------------------------------
function saveState(fileName, net, state)
% -------------------------------------------------------------------------
save(fileName, 'net', 'state') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
if exist(fileName)
  save(fileName, 'stats', '-append') ;
else
  save(fileName, 'stats') ;
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'stats') ;
net = vl_simplenn_tidy(net) ;
if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
%clear vl_tmove vl_imreadjpeg ;
disp('Clearing mex files') ;
clear mex ;
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(params, cold)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end
end
if numGpus >= 1 && cold
  fprintf('%s: resetting VLDT_GPU\n', mfilename) ;
  clearMex() ;
  if numGpus == 1
    disp(gpuDevice(params.gpus)) ;
  else
    spmd
      clearMex() ;
      disp(gpuDevice(params.gpus(labindex))) ;
    end
  end
end
