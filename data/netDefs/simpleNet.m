function [net, net_seg, opts]=netStructure(opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%DESIGN THE NETWORK 6%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Original dataset
opts.origDatasetDir='data/db_orig/'; %Folder that contains the dataset
opts.origDatasetIdx='ISIC_2017.mat'; %Indexation File

%Image size to be used in the CNN
opts.imSize=[256 256];
%DB name and path
opts.dbName='melanomas';
opts.imdbFolder = fullfile('data', ['imdb_' opts.dbName '_' num2str(opts.imSize(1)) 'x' num2str(opts.imSize(2))]);

%We use sets 1 to train
opts.trainDB=1;
%We just use the challenge to validation 
opts.valDB=2;

opts.imdbPath= fullfile('data', ['imdb_' opts.dbName '_' num2str(opts.imSize(1)) 'x' num2str(opts.imSize(2)) '_tr_' num2str(opts.trainDB) '_val_' num2str(opts.valDB) '.mat']);
%Network Training options
lr=logspace(-3,-4,10); %Learning rate per epoch
opts.train.learningRate = lr ; %Learning rate
opts.train.numEpochs = numel(lr) ;
opts.train.batchSize = 256 ; %Batch size
opts.train.numSubBatches = 4 ;
opts.train.weightDecay = 0.0005 ; %Weight decay in learning function (regularization)
opts.train.momentum = 0.9 ; %Momentum parameter
opts.train.gpus = 1; %Leave empty if no GPUs are available
opts.train.errorFunction = 'auc' ;
opts.train.nesterovUpdate = true ; %Using Nesterov moments

%In this case we don't use net_seg
net_seg=[];

%We do fine-tuning over res-net
net=load('models/imagenet-resnet-50-dag.mat');
if isfield(net, 'net') ;
  net = net.net ;
end
isDag = isfield(net, 'params') ;
if isDag
  opts.networkType = 'dagnn' ;
  net = dagnn.DagNN.loadobj(net) ;
end

%Net definition
net.meta=[];
net.meta.inputs.name='data';
net.meta.inputs.size=[opts.imSize(1) opts.imSize(2) 3 opts.train.batchSize];
net.meta.classes.name={'benign','melanoma','seborrheic keratosis'};
net.meta.classes.description={'benign nevus','malignant melanoma','seborrheic keratosis'};    


%%%%%%%%%%%%%%%%%%%%%REMOVE IMAGENET LAST LAYERS%%%%%%%%%%%%%%%%%%
%We remove the last layers that are for Imagenet
net.removeLayer('prob'); %Output
net.removeLayer('fc1000'); %Fully-connected
net.removeLayer('pool5'); %Pooling

%New pooling
net.addLayer('pool5' , ...
             dagnn.Pooling('poolSize', [8 8], 'method', 'avg'), ...
             'res5cx', ...
             'pool5') ;
         
%Generate the last fc layer to generate the three outputs
net.addLayer('prediction' , ...
             dagnn.Conv('size', [1 1 2048 3]), ...
             'pool5', ...
             'prediction', ...
             {'prediction_f', 'prediction_b'}) ;
         
%Add layers for cost and AUC performance metric
net.addLayer('loss', ...
             dagnn.Loss('loss', 'softmaxlog') ,...
             {'prediction', 'label'}, ...
             'objective') ;

net.addLayer('performance', ...
             dagnn.Performance('metric', 'auc') ,...
             {'prediction', 'label'}, ... %inputs
             'auc') ; %outputs   
         
% Make sure that the input is called 'input'
v = net.getVarIndex('data') ;
if ~isnan(v)
    net.renameVar('data', 'input') ;
end
% Init empty parameters randomly
net.initParams();


end




