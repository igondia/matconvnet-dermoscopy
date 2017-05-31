function [net, stats] = cnn_train_mod(net, net_seg, imdb, getBatch, varargin)
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
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;
opts.numArch = 1 ;
opts.useGpu =  false;
opts.GrayScale = 0 ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = 'ap' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.plotStatistics = true;
opts.perf_data=struct('labels',[],'predictions',[],'perf',[]);
opts.validLabelsError = 1;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
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

% if ~evaluateMode
%   for i=1:numel(net.layers)
%     if isfield(net.layers{i}, 'weights'); 
%       J = numel(net.layers{i}.weights) ;
%       if ~isfield(net.layers{i}, 'learningRate') 
%         net.layers{i}.learningRate = ones(1, J) ;
%       end
%       if ~isfield(net.layers{i}, 'weightDecay')
%         net.layers{i}.weightDecay = ones(1, J) ;
%       end
%       if i<(numel(net.layers)-opts.backPropDepth)
%           net.layers{i}.learningRate = zeros(1, J) ;
%       end
%       for j=1:J
%           if net.layers{i}.learningRate(j)>0
%             net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
%           end
%       end
%     end
%   end
% end

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

  if numel(params.gpus) <= 1
    [net, state] = processEpoch(net, net_seg, state, params, 'train') ;
    [net, state] = processEpoch(net, net_seg, state, params, 'val') ;
    if ~evaluateMode
      saveState(modelPath(epoch), net, state) ;
    end
    lastStats = state.stats ;
  else
    spmd
      [net, state] = processEpoch(net, net_seg, state, params, 'train') ;
      [net, state] = processEpoch(net, net_seg, state, params, 'val') ;
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
      fieldnames(stats.val)'), {'num', 'time','aux_auc'},'stable') ;
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
      legend(leg{:},'Location','northeast') ;
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
function err = error_auc(params, labels, res)
% -------------------------------------------------------------------------
%Si solo hay una salida

if(size(res(end-1).x,3)==1)
    predictions=squeeze(gather(res(end-1).x));
    posClass=1;
    labels=labels';
%Si tenemos mas de una    
else
    predictions= squeeze(gather(vl_nnsoftmax(res(end-1).x,[])));
    %Remove the first column as is the background
    predictions=predictions(2:end,:)';
    labels=(labels-1)';
end
tlabels=[params.perf_data.labels;labels];
tpredictions=[params.perf_data.predictions;predictions];
[numCat]=size(predictions,2);
auc=zeros(1,numCat);
for c=1:numCat
    labels_aux=tlabels==c;
    posClass=1;
    if(sum(labels_aux)>0)

        %[X,Y,T,auc1] = perfcurve(labels_aux,tpredictions(:,c),posClass);
	auc(c)=fAUC(labels_aux>0,tpredictions(:,c));
	%[auc1 auc(c)]
    else
        auc(c)=0.5;
    end
end
err.labels = labels;
err.predictions = predictions;
err.perf = auc*length(labels);

% -------------------------------------------------------------------------
function err = error_ap(params, labels, res)
% -------------------------------------------------------------------------
%Si solo hay una salida
if(size(res(end-1).x,3)==1)
    predictions=squeeze(gather(res(end-1).x));
    labels=labels(params.validLabelsError>0,:)';
    labels(labels<0)=0;

else
    predictions= vl_nnsoftmax(res(end-1).x,[]);
    predictions = squeeze(predictions(:,:,2,:));
    labels=labels';
    labels(labels==1)=0;
    labels(labels==2)=1;
end
    

tlabels=[params.perf_data.labels;labels];
tpredictions=[params.perf_data.predictions;predictions];
numPat=size(tlabels,2);
if(numPat>1)
    ap=zeros(1,numPat);
    for i=1:1:numPat
        ap(i) = computeAP(tpredictions(:,i),tlabels(:,i)); 
        if(isnan(ap(i)))
            disp('error ap nan');
            ap(i)=0;
        end
    end
else
    ap = computeAP(tpredictions,tlabels); 
end
err.labels = labels;
err.predictions = predictions;
err.perf = ap*length(predictions);

% -------------------------------------------------------------------------
function err = error_none(params, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, net_seg, state, params, mode)
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
  net_seg = vl_simplenn_move(net_seg, 'gpu') ;
  one = gpuArray(single(1)) ;
  for i = 1:numel(state.momentum)
    for j = 1:numel(state.momentum{i})
      state.momentum{i}{j} = gpuArray(state.momentum{i}{j}) ;
    end
  end
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
res_seg = [] ;
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

    bdata = params.getBatch(params.imdb, batch);
    im=bdata{1};
    pcoords=bdata{2};
    labels=bdata{3};
    instanceWeights=bdata{4};
    
    
    if params.prefetch
      if s == params.numSubBatches
        batchStart = t + (labindex-1) + params.batchSize ;
        batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
      params.getBatch(params.imdb, nextBatch,'simplenn') ;
    end

    if numGpus >= 1
      im = gpuArray(im) ;
    end

    if strcmp(mode, 'train')
      dzdy = one ;
      evalMode = 'normal' ;
    else
      dzdy = [] ;
      evalMode = 'test' ;
    end
    
    % evaluate the CNN
    net.layers{end}.opts=[];
    %Binary Labels
    if(size(net.layers{end-1}.weights{1},4)==1)
     net.layers{end}.opts.loss='logistic';
    %Category labels 
    else
        net.layers{end}.opts.loss='softmaxlog';
    end
    %Weights for AUC
    net.layers{end}.class = reshape(labels,[1 1 size(labels,1) size(labels,2)]) ;
    net.layers{end}.opts.instanceWeights=reshape(instanceWeights,[1 1 size(instanceWeights,1) size(instanceWeights,2)]);
    
    
    %First, execute the net_seg
    res_seg = vl_simplenn_mask(net_seg, im, pcoords, [], res_seg, ...
                      'accumulate', s ~= 1, ...
                      'mode', 'test', ...
                      'conserveMemory', params.conserveMemory, ...
                      'backPropDepth', 1, ...
                      'sync', params.sync, ...
                      'cudnn', params.cudnn, ...
                      'parameterServer', parserv, ...
                      'holdOn', s < params.numSubBatches) ;
                  
    %Quit background
    wmod=res_seg(end).x(:,:,2:end,:);
    %Reduce analysis to lession masks
    masks=pcoords(:,:,1,:)>0;
    masks = imresize(masks,[size(wmod,1) size(wmod,2)],'Method','nearest');
    wmod=bsxfun(@times,wmod,masks);
    
    
    res_seg=[];
    for l=1:length(net.layers)
        %In case of modulation, add the whole image
        if(strcmp(net.layers{l}.type,'modulateInputs'))
            %Sin máscara
            wmod = cat(3,ones(size(wmod,1),size(wmod,2),1,size(wmod,4),'single'),wmod);
            %Con máscara
%             wmod = cat(3,masks,wmod);
            net.layers{l}.wmod=wmod;
        elseif(strcmp(net.layers{l}.type,'fuseInputs'))
            net.layers{l}.wmod=wmod;
            %We normalize things to become zero mean and 1 std deviation 
            %mult=size(net.layers{l-2}.weights{1},3)/size(wmod,3);
            net.layers{l}.wmod=vl_nnbnorm(net.layers{l}.wmod,gpuArray(ones(size(wmod,3), 1, 'single')), gpuArray(zeros(size(wmod,3), 1, 'single')),'epsilon', 1e-4) ;
        end
    end
    clear wmod;


    res = vl_simplenn_mask(net, im, pcoords, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'mode', evalMode, ...
                      'conserveMemory', params.conserveMemory, ...
                      'backPropDepth', params.backPropDepth, ...
                      'sync', params.sync, ...
                      'cudnn', params.cudnn, ...
                      'parameterServer', parserv, ...
                      'holdOn', s < params.numSubBatches) ;

    % accumulate training errors
    auxerr = params.errorFunction(params, labels, res);
    %AUC or AP
    if(isstruct(auxerr))
        params.perf_data.labels=[params.perf_data.labels;auxerr.labels];
        params.perf_data.predictions=[params.perf_data.predictions;auxerr.predictions];
        params.perf_data.perf=[params.perf_data.perf;auxerr.perf];
        mean_perf=params.perf_data.perf(end,:);
%         error(2)=0;
        error = sum([error, [sum(double(gather(res(end).x))) ;reshape(mean(mean_perf),[],1) ; reshape(mean_perf,[],1) ]],2) ;
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
  for f = setdiff(fieldnames(stats)', {'num', 'time'},'stable') ;
    f = char(f) ;
    fprintf(' %s: ', f);
    for j=1:length(stats.(f))
        fprintf(' %.3f', stats.(f)(j)) ;
    end
  end
  fprintf('\n') ;
%fprintf('Time seg %f diag %f auc %f grad %f stats %f\n',ts(1),ts(2),ts(3),ts(4),ts(5));

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
        try
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, net.layers{l}.weights{j}) ;
        catch
            keyboard;
        end
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
if(length(errors)>numel(params.errorLabels)+1)
    stats.(['aux_' params.errorLabels{end}])=errors(numel(params.errorLabels)+2:end)';
end

% -------------------------------------------------------------------------
function saveState(fileName, net, state)
% -------------------------------------------------------------------------
save(fileName, 'net', 'state','-v7.3') ;

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

function [auc,fpr,tpr] = fAUC(labels,scores)
% function [auc,fpr,tpr] = myauc(labels,scores)
%
% This function calculates m AUC values for m ranked lists.
% n is the number of ranked items. 
% m is the number of different rankings.
%
% Input:  labels is nXm binary logical.
%         scores is nXm real. For a high AUC the higher scores should have
%         labels==1.
%         plot_flag: binary flag, if TRUE then m ROC curves will be plotted
%         (default FALSE).
%
% Output: auc is mX1 real, the Area Under the ROC curves.
%         fpr is nXm real, the false positive rates.
%         tpr is nXm real, the true positive rates.
plot_flag=0;
if ~exist('plot_flag','var')
    plot_flag = 0;
end
if ~islogical(labels)
    error('labels input should be logical');
end
if ~isequal(size(labels),size(scores))
    error('labels and scores should have the same size');
end
[n,m] = size(labels);
num_pos = sum(labels);
if any(num_pos==0)
    error('no positive labels entered');
end
if any(num_pos==n)
    error('no negative labels entered');
end

[~,scores_si] = sort(scores,'descend');
clear scores
scores_si_reindex = scores_si+ones(n,1)*(0:m-1)*n;
l = labels(scores_si_reindex);
clear scores_si labels 

tp = cumsum(l==1,1);
fp = repmat((1:n)',[1 m])-tp;

num_neg = n-num_pos;
fpr = bsxfun(@rdivide,fp,num_neg); %False Positive Rate
tpr = bsxfun(@rdivide,tp,num_pos); %True Positive Rate

%Plot the ROC curve
if plot_flag==1
    plot(fpr,tpr);
    xlabel('False Positive');
    ylabel('True Positive');
end

auc = sum(tpr.*[(diff(fp)==1); zeros(1,m)])./num_neg;
