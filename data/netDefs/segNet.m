function [net opts]=netStructure(opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%DESIGN THE NETWORK 6%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.imSize=[256 256];

%Original dataset
opts.origDatasetDir='data/db_orig/'; %Folder that contains the dataset
opts.origDatasetIdx='ISIC_2016.mat'; %Indexation File

%Image size to be used in the CNN
opts.imSize=[256 256];
%DB name and path
opts.dbName='ISIC_2016';
opts.imdbFolder = fullfile('data', ['imdb_' opts.dbName '_' num2str(opts.imSize(1)) 'x' num2str(opts.imSize(2))]);


%We just use the 2016 ISIC challenge to train => We need weak
%annotations!!!
%We use sets 1 to train
opts.trainDB=1;
%Don't use anaything to validate
opts.valDB=[];

%We use the whole sets
opts.numSet=0;
opts.imdbPath= fullfile('data', ['imdb_' opts.dbName '_' num2str(opts.imSize(1)) 'x' num2str(opts.imSize(2)) '_tr_' num2str(opts.trainDB) '_val_' num2str(opts.valDB) '.mat']);

%Network Training options
lr=0.01*ones(1,200); %Learning rate per epoch
opts.train.learningRate = lr ; %Learning rate
opts.train.numEpochs = numel(lr) ;
opts.train.batchSize = 256 ; %Batch size
opts.train.numSubBatches = 32 ;
opts.train.weightDecay = 0.0005 ; %Weight decay in learning function (regularization)
opts.train.momentum = 0.9 ; %Momentum parameter
opts.train.gpus = 1; %Leave empty if no GPUs are available
opts.train.errorFunction = 'auc' ;
opts.train.nesterovUpdate = true ; %Using Nesterov moments
opts.networkType = 'dag' ;


%%%%%%%%COMPLEJIDAD%%%%%%%%%%%%%%%%%%%
% Define a network similar to LeNet
f=1/100 ;
net.layers = {} ;

net=load('models/imagenet-resnet-50-dag.mat');
if isfield(net, 'net') ;
  net = net.net ;
end

isDag = isfield(net, 'params') ;

if isDag
  opts.networkType = 'dagnn' ;
  net = dagnn.DagNN.loadobj(net) ;
end

%%%%%%%%%%%%%%%%%%%%%We remove strides%%%%%%%%%%%%%%%%%%
for l=2:length(net.layers)
    if(isa(net.layers(l).block,'dagnn.Conv'))
        if(sum(net.layers(l).block.stride)>2)
            net.layers(l).block.stride=[1 1];
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Setting-up Weak learning%%%%%%%%%%%%%%%%%%%%%%%
net.removeLayer('fc1000'); 
net.removeLayer('prob'); 
net.removeLayer('pool5'); 

%Final prediction with 9 patterns: background and the rest
net.addLayer('prediction' , ...
             dagnn.Conv('size', [1 1 2048 9]), ...
             'res5cx', ...
             'prediction', ...
             {'prediction_f', 'prediction_b'}) ;

%Setting up the weak loss (A and b set the constraints in the inequalities)
%Explanation:
    % Each label value contains two constraints:
    %  - If label=0 the dermoscopic feature is not present. 
    %       Constraint 1: the accumulated prob is lower than 0.01
    %       Constraint 2: unused
    %  - If label=1 the dermoscopic feature is local
    %       Constraint 1: the accumulated probability is higher than 0.15
    %       Constraint 2: the accumulated probability is lower than 0.5
    %  - If label=2 the dermoscopic feature is global
    %       Constraint 1: the accumulated probability is higher than 0.50
    %       Constraint 2: unused
    %  - If label=3 the dermoscopic feature is only in lesion borders
    %       Constraint 1: the accumulated probability is higher than 0.05
    %       Constraint 2: the accumulated probability is lower than 0.5
%beta sets the weights for each dermoscopic feature considering an unbalanced problem
%Explanation: beta has length of the number of dermoscopic features +1 (the
%background/skin).
    %Dermoscopic features
    %0 'background/skin';
    %1 'globular/cobblestone';
    %2 'homogeneous';
    %3 'reticular';
    %4 'streaks';
    %5 'blueveil';
    %6 'vascularStr';
    %7 'hypopigmentation';
    %8 'regressionStr';
    %9 'unspecific';
%The weights are the proportion of images containing the dermoscopic feature    
net.addLayer('weakloss' , ...
             dagnn.WeakLoss('A', single([   -1 0    1   -1    1  0    1    -1]),...
                'b',single(             [-0.01 0 0.15 -0.5 0.50  0 0.05 -0.50]), ...
                'beta',single([1.0 0.6306    0.3556    0.7568    0.0721    0.0557    0.1204    0.2597    0.4915]), ...
					      'maxLambda',1000.00, ...
					      'gamma',1.0),...
             {'prediction', 'label'},...
             'objective',...
             'pcoords');

%Performance metric
net.addLayer('performance', ...
             dagnn.SegPerformance('metric', 'wseg_bal'),...
             {'prediction', 'label'}, ... %inputs
             'wseg_bal',...
             'pcoords'); %params)

%We do not consider the dermoscopic feature 7: Hypopigmentation
%We also remove the first label in the annotation vector as it is the
%diagnosis
opts.weightPatterns=[0 1 1 1 1 1 1 0 1 1];
opts.validLabelsError=[0 1 1 1 1 1 1 0 1 1];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make sure that the input is called 'input'
v = net.getVarIndex('data') ;
if ~isnan(v)
    net.renameVar('data', 'input') ;
end
% Init parameters randomly
net.initParams();

         
pindex=net.getParamIndex('pcoords');
net.params(pindex).learningRate=0;
net.params(pindex).trainMethod='notrain';

pindex=net.getParamIndex('prediction_f');
net.params(pindex).value(:)=1/2048;


