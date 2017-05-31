function [net, info] = cnn_train_weakLatent(net, imdb, getBatch, varargin)
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
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
opts.numArch = 1 ;
opts.GrayScale = 0 ;
opts.useGpu =  false;
opts.numSet =  1;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = false ;
opts.errorFunction = 'weak_seg' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.plotStatistics = true;
opts.perf_data=struct('labels',[],'predictions',[],'perf',[]);
opts.validLabelsError = 1;
opts = vl_argparse(opts, varargin) ;


if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
% if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
% if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = vl_simplenn_tidy(net); % fill in some eventually missing values
net.layers{end-1}.precious = 1; % do not remove predictions, used for error
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;


 
evaluateMode = isempty(opts.train) ;

if ~evaluateMode
  for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single') ;
      end
    end
  end
end

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

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  load(modelPath(start), 'net', 'info') ;
  net = vl_simplenn_tidy(net) ; % just in case MatConvNet was updated
else
  save(modelPath(0), 'net') ;
end

for epoch=start+1:opts.numEpochs

  % train one epoch and validate
  learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  train = opts.train(randperm(numel(opts.train))) ; % shuffle
  val = opts.val;

  if numGpus <= 1
    [net,stats.train,prof] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net) ;
    if(~isempty(val))
        [~,stats.val] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net) ;
    end
    if opts.profile
      profile('viewer') ;
      keyboard ;
    end
  else
    fprintf('%s: sending model to %d GPUs\n', mfilename, numGpus) ;
    spmd(numGpus)
      [net_, stats_train_,prof_] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net) ;
      [~, stats_val_] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net_) ;
    end
    net = net_{1} ;
    stats.train = sum([stats_train_{:}],2) ;
    stats.val = sum([stats_val_{:}],2) ;
    if opts.profile
      mpiprofile('viewer', [prof_{:,1}]) ;
      keyboard ;
    end
    clear net_ stats_train_ stats_val_ ;
  end

  % save
  if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
  for f = sets
    f = char(f) ;
    n = numel(eval(f)) ;
    info.(f).speed(epoch) = n / stats.(f)(1) * max(1, numGpus) ;
    info.(f).objective(epoch) = stats.(f)(2) / n ;
    info.(f).error(:,epoch) = stats.(f)(3) / n ;
  end
  if ~evaluateMode
    fprintf('%s: saving model for epoch %d\n', mfilename, epoch) ;
    tic ;
    save(modelPath(epoch), 'net', 'info') ;
    fprintf('%s: model saved in %.2g s\n', mfilename, toc) ;
  end

  if opts.plotStatistics
    switchfigure(1) ; clf ;
    subplot(1,1+hasError,1) ;
    if ~evaluateMode
      semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
      hold on ;
    end
    semilogy(1:epoch, info.val.objective, '.--') ;
    xlabel('training epoch') ; ylabel('energy') ;
    grid on ;
    h=legend(sets) ;
    set(h,'color','none');
    title('objective') ;
    if hasError
      subplot(1,2,2) ; leg = {} ;
      if ~evaluateMode
        plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
        hold on ;
        leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
      end
      plot(1:epoch, info.val.error', '.--') ;
      leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
      set(legend(leg{:},'Location','south'),'color','none') ;
      grid on ;
      xlabel('training epoch') ; ylabel('perf') ;
      title('error') ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
  end
end

% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res,pcoords)
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
function err = error_binary(opts, labels, res,pcoords)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
predictions = squeeze(predictions(:,:,2,:));
labels_signed=labels;
labels_signed(labels==1)=-1;
labels_signed(labels==2)=1;
error = bsxfun(@times, predictions, labels_signed) < 0 ;
err = sum(error(:)) ;

% -------------------------------------------------------------------------
function err = error_auc(opts, labels, res,pcoords)
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

tlabels=[opts.perf_data.labels;labels];
tpredictions=[opts.perf_data.predictions;predictions];
[X,Y,T,auc] = perfcurve(labels,predictions,posClass);
err.labels = tlabels;
err.predictions = tpredictions;
err.perf = auc;

% -------------------------------------------------------------------------
function err = error_ap(opts, labels, res, pcoords)
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
    predictions=squeeze(gather(res(end-1).x(:,:,opts.validLabelsError>0,:)))';
end
labels=labels(opts.validLabelsError>0,:)';
labels(labels<0)=0;
predictions=predictions(:,2:end);
labels=labels(:,2:end);
tlabels=[opts.perf_data.labels;labels];
tpredictions=[opts.perf_data.predictions;predictions];
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
function err = error_wseg(opts, labels, res, pcoords)
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
    predictions=squeeze(gather(res(end-1).x(:,:,opts.validLabelsError>0,:)))';
end
labels=labels(opts.validLabelsError>0,:)';
labels(labels<0)=0;
%Now we can decide the final predictions based on the A,b
label_pred=zeros(size(labels));
label_pred(predictions<=opts.b(1))=0;
label_pred(predictions>=opts.b(3) & predictions<=opts.b(4))=1;
label_pred(predictions>=opts.b(5))=2;

tlabels=[opts.perf_data.labels;labels];
tpredictions=[opts.perf_data.predictions;label_pred];
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
function err = error_wseg_bal(opts, labels, res, pcoords)
% -------------------------------------------------------------------------
%Si solo hay una salida
numCat=size(res(end-1).x,3);
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
    predictions=squeeze(gather(res(end-1).x(:,:,opts.validLabelsError>0,:)))';
end
labels=labels(opts.validLabelsError>0,:)';
labels(labels<0)=0;
%Now we can decide the final predictions based on the A,b
label_pred=ones(size(labels));
minVal=0.05;%mean(opts.b([1 3]));
label_pred(predictions<=minVal)=0;
% label_pred(predictions>=opts.b(3) & predictions<=opts.b(4))=1;
% label_pred(predictions>=opts.b(5))=2;
[sum(label_pred);sum(labels>0)]

%Remove the BG from the analysis
labels=labels(:,2:end);
predictions=predictions(:,2:end);
%Check how many conditions we fulfill
tlabels=[opts.perf_data.labels;labels];
tpredictions=[opts.perf_data.predictions;predictions];
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
err.perf = acc;
% -------------------------------------------------------------------------
function err = error_none(opts, labels, res,pcoords)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function  [net_cpu,stats,prof] = process_epoch(opts, getBatch, epoch, subset, learningRate, imdb, net_cpu)
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
% if(numel(subset)>1024)
%     subset=subset(1:1024);
% end

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
    
% batch=1:24:opts.batchSize*24;    
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

    
   %Convert labels to single
    net.layers{end}.labels = single(labels) ;
    
    if(~isfield(net.layers{end}, 'lambda'))
        net.layers{end}.lambda=zeros([2*size(labels,1) size(labels,2)],'single');
    end
    if numGpus >= 1
      net.layers{end}.lambda=gpuArray(net.layers{end}.lambda);
      net.layers{end}.labels=gpuArray(net.layers{end}.labels);
    end
    
    if training, dzdy = one; else, dzdy = [] ; end
    [res net]= vl_simplenn_mask(net, im, pcoords, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'mode', evalMode, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync, ...
                      'cudnn', opts.cudnn) ;

    
    % accumulate training errors
    auxerr = opts.errorFunction(opts, labels, res,pcoords);
    %AUC or AP
    if(isstruct(auxerr))
        opts.perf_data.labels=[opts.perf_data.labels;auxerr.labels];
        opts.perf_data.predictions=[opts.perf_data.predictions;auxerr.predictions];
        opts.perf_data.perf=[opts.perf_data.perf;auxerr.perf];
        mean_perf=opts.perf_data.perf(end,:);
%         error(2)=0;
        error = sum([error, [sum(double(gather(res(end).x))) ;reshape(mean_perf(1),[],1) ; reshape(mean_perf(2:end),[],1) ]],2) ;
    else
        error = sum([error, [sum(double(gather(res(end).x))) ;reshape(auxerr,[],1) ; ]],2) ;
    end

    numDone = numDone + numel(batch) ;
  end % next sub-batch

  % gather and accumulate gradients across labs
  if training
    if numGpus <= 1
      [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res, mmap) ;
    end
  end

  % collect and print learning statistics
  time = toc(start) ;
  stats = sum([stats,[0 ; error]],2); % works even when stats=[]
  
  stats(1) = time ;
  n = t + batchSize - 1 ; % number of images processed overall
  speed = n/time ;
  fprintf('%.1f Hz%s\n', speed) ;
    
  m = n / max(1,numlabs) ; % num images processed on this lab only
  %We set this error
  if(~isempty(opts.perf_data.perf))
      stats(3:end) = error(2:end)*m;
  end
  fprintf(' obj:%.3f', stats(2)/m) ;
  numExtraPerf=length(error)-2;
  %Printing average results
  for i=1:numel(opts.errorLabels)
    fprintf(' m-%s:%.3f', opts.errorLabels{i}, stats(i+2)/m) ;
%     fprintf(' g-%s:%.3f', opts.errorLabels{i}, mean(stats(4:end))/m) ;
    
    if(numExtraPerf>0)
        fprintf(' e-%s',opts.errorLabels{i});
        for j=1:numExtraPerf
            fprintf(' %.2f', stats(j+3)/m) ;
        end
    end
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
% fprintf('\n');
refVal=-1;
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
      %if(strcmp(net.layers{l}.type, 'conv') && j==1)
      %  if(refVal<0)
      %      refVal=0.1*mean(abs(res(l).dzdw{j}(:)));
      %  end
      %  net.layers{l}.learningRate(j)=(refVal/mean(abs(res(l).dzdw{j}(:))));
      %  net.layers{l}.learningRate(j+1)=net.layers{l}.learningRate(j);
      %end
      
      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = lr * net.layers{l}.learningRate(j) ;
      net.layers{l}.momentum{j} = ...
        opts.momentum * net.layers{l}.momentum{j} ...
        - thisDecay * net.layers{l}.weights{j} ...
        - (1 / batchSize) * res(l).dzdw{j} ;
       net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
            thisLR * net.layers{l}.momentum{j} ;
%     if(strcmp(net.layers{l}.type,'conv') && j==1)
%         
%         disp(num2str([l mean(abs(opts.momentum * net.layers{l}.momentum{j}(:))) -mean(abs(thisDecay * net.layers{l}.weights{j}(:))) -mean(abs((1 / batchSize) * res(l).dzdw{j}(:)))]));
%         prevW=net.layers{l}.weights{j};
%      
%         disp(num2str([mean(abs(prevW(:))) thisLR mean(abs(thisLR * net.layers{l}.momentum{j}(:))) mean(abs(net.layers{l}.weights{j}(:)))]));
%         
%     end
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
% pause;
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


