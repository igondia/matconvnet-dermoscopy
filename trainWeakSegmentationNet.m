%Function that trains a weak segmentation network
%       Inputs:
%            modelStructurePath: path to the .m defining the network
%            expDir: Folder where storing the results of the training
%            process
%            opts [optional]: opts to be included in the training process
function trainWeakSegmentationNet(modelStructurePath,expFolder,varargin)
%setenv('LD_LIBRARY_PATH','/usr/local/cuda-7.5/lib64:/usr/lib64/'); 
%Path to the needed files
addpath('examples/melanomas');
dataDir='./data/';
run('matlab/vl_setupnn.m') ;

opts.expDir = expFolder;
%Path where the original dataset is stored
opts.origDatasetDir='db';
%Indextaion File
opts.origDatasetIdx=['db.idx'];
%Path to the cnn dataset 
opts.imdbPath = fullfile('data', 'imdb_.mat');

opts.train.batchSize = 256 ;
opts.train.numEpochs = 25 ;
opts.train.continue = true ;
opts.train.gpus = [];
opts.train.learningRate=logspace(-1, -4, 20) ;
opts.train.levels_update=-1;
opts.networkType='dag';

opts.batchNormalization = false ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightPatterns = 0.5;
opts.train.errorFunction = 'weak_seg' ;
opts.numSet=0;
opts.numSets=10;
opts.dbName='ISIC_2016';


%We just use the challenge to train
opts.trainDB=1;
%We just use the challenge to test
opts.valDB=2;
% opts.train.plotDiagnostics = 1;
opts = vl_argparse(opts, varargin) ;
opts.train.expDir = opts.expDir ;
if(~exist(opts.expDir,'dir'))
    mkdir(opts.expDir);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%STEP 1: DESIGN THE NETWORK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%We read netStructure in the data folder and override options
[folder, functionName, ext]=fileparts(modelStructurePath);
addpath(folder);
%Read the network
[net, opts]=feval(functionName,opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%STEP 2: READ THE DATASET%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(~exist(opts.imdbPath,'file'))
    aux=load([opts.origDatasetDir '/' opts.origDatasetIdx]);
    names=fieldnames(aux);
    db=aux.(names{1});
    imdb = getMelanomasImdb(db,opts.origDatasetDir,opts.imSize,opts.imdbFolder,opts.trainDB,opts.valDB);
    save(opts.imdbPath,'imdb');
end
load(opts.imdbPath,'imdb');
%Set normalization
net.meta.normalization.vdata_mean=imdb.images.vdata_mean;
net.meta.normalization.size=net.meta.inputs.size;

%%%%%%%%%%%%%%%%%%%Prepare the labels%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nIms=size(imdb.images.labels,1);
imdb.images.instanceWeights=[];
%We remove non-interesting labels
idxUtiles=opts.weightPatterns>0;
imdb.images.idxUtiles=idxUtiles;
imdb.images.labels=imdb.images.labels(:,idxUtiles);
opts.validLabelsError=opts.validLabelsError(idxUtiles);
imdb.images.instanceWeights=ones(size(imdb.images.labels));
opts.weightPatterns=opts.weightPatterns(idxUtiles);
imdb.images.instanceWeights=imdb.images.instanceWeights.*repmat(opts.weightPatterns,nIms,1);
%Change streaks (pattern 4) to label 3
imdb.images.labels(imdb.images.labels(:,4)>0,4)=3;
imdb.images.labels=[zeros(nIms,1) imdb.images.labels];
opts.validLabelsError=[1 opts.validLabelsError];
imdb.images.instanceWeights=[ones(nIms,1) imdb.images.instanceWeights];  
%Set Background to label 3 (only in borders as it represents skin)
imdb.images.labels(:,1)=3;
opts.train.validLabelsError=opts.validLabelsError;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%TRAIN THE NETWORK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train_mod ;
  case 'dagnn', trainFn = @cnn_train_dag_weakLatent ;
end
trainidx=find(imdb.images.set == 1);
validx=find(imdb.images.set == 2);
[net, info] = trainFn(net, imdb, @getBatch, opts.train, 'train', trainidx,'val',validx) ;

% --------------------------------------------------------------------
function varargout = getBatch(imdb, batch, networkType,opts)
% --------------------------------------------------------------------
vdata_mean=imdb.images.vdata_mean;
labels = imdb.images.labels(batch,:)' ;
paths=imdb.images.paths;
im=[];
pcoords=[];
numIm=length(batch);
%We read the first one
load(paths{batch(1)},'data','pcoord');
[H W C]=size(data);
im=single(zeros(H,W,C,numIm));
pcoords=single(zeros(H,W,2,numIm));
im(:,:,:,1)=single(data);
pcoords(:,:,:,1)=pcoord;

for i=2:numIm
    aux=load(paths{batch(i)},'data','pcoord');
    im(:,:,:,i)=single(aux.data);
    pcoords(:,:,:,i)=aux.pcoord;
end

im=bsxfun(@minus,im,vdata_mean);

switch networkType
    case 'simplenn'
      varargout = {im, pcoords,labels};%,instanceWeights} ;
    case 'dagnn'
      varargout{1} = {'input', im, 'label', labels,'pcoords',pcoords};%,'instanceWeights',instanceWeights} ;
end
  
% --------------------------------------------------------------------
function imdb = getMelanomasImdb(idxStruct,origDatasetDir,imSize,dbFolder,trainDB,valDB)
% --------------------------------------------------------------------
imDBFolder=[dbFolder '/db_images'];
mkdir(imDBFolder);
%Icons folder
imDBIFolder=[dbFolder '/db_images_icons'];
mkdir(imDBIFolder);


%Get the valid sets
sets=cat(1,idxStruct.db);
idxTrain=find(sets==trainDB);
if(~isempty(valDB))
    idxVal=find(sets==valDB);
else
    idxVal=[];
end
idxTotal=[idxTrain;idxVal];

numTrain=length(idxTrain); %idx of train images
numVal=length(idxVal); %idx of val images
numImages=numTrain+numVal; %Number of total images

%numCategories: 1 diagnosis + 9 structural patterns
numCat=10;

%Parameters of the data augmentation
numAngles=6;
numCrops=4;
numVariations=numAngles*numCrops;
numTrainTotal=numTrain*numVariations;

%Training and validation dataset
parfor f=1:numImages
    idxIm=idxTotal(f);
    [result numDoneImages]=unix(['ls -1 ' imDBFolder '/' num2str(f) '/*.mat | wc -l']);
    numDoneImages=str2num(numDoneImages);
    if(numDoneImages>=numVariations)
        disp(['Image ' num2str(idxIm) '/' num2str(numImages) ' was done']);
    else
        if(~exist([imDBFolder '/' num2str(f)],'dir'))
            mkdir([imDBFolder '/' num2str(f)]);
            mkdir([imDBFolder '_icons/' num2str(f)]);
        end
        disp(['Computing image ' num2str(idxIm) '/' num2str(numImages)]);
        
        %Reading image
        im=imread([origDatasetDir '/' idxStruct(idxIm).impath]);
        %Reading mask
        mask=imread([origDatasetDir '/' idxStruct(idxIm).maskpath])>128;
        %Data Augmentation
        [im,pcoords]=dataAugmentation(im,mask,imSize,numAngles,numCrops);
        %Save the variations
        for v=1:numVariations
            data=im(:,:,:,v);
            pcoord=single(pcoords(:,:,:,v));
            imgPath=[imDBFolder '/' num2str(f) '/' num2str(v) '.mat'];
            imgIPath=[imDBIFolder '/' num2str(f) '/' num2str(v) '.jpg'];
            saveImage(imgPath,data,pcoord);
            imwrite(imresize(uint8(data),0.10),imgIPath);
        end
        
    end
end

%Computing the mean of the images just over the training dataset
dataMeanFile=[imDBFolder '/dataMeanLabels.mat'];
if(exist(dataMeanFile,'file'))
    load(dataMeanFile, 'dataMean','vdataMean');
else
    dataMean=zeros(imSize(1),imSize(2),3);
    for f=1:numTrain
        disp(['Gathering mean from doc ' num2str(f) '/' num2str(numImages)])
        for v=1:numVariations
            imgPath=[imDBFolder '/' num2str(f) '/' num2str(v) '.mat'];
            aux=load(imgPath,'data');
            data=aux.data;
            dataMean=dataMean+double(data);
        end
    end
    dataMean=single(dataMean/numTrainTotal);
    vdataMean=mean(mean(dataMean,2),1);
    if(sum(isinf(dataMean(:))))
        disp('Overflow in dataMean');
    end
    save([imDBFolder '/dataMeanLabels.mat'], 'dataMean','vdataMean');
end

%Data indexation
numTrainTotal=numVariations*numTrain; %Total number of train images (with data augmentation)
numFinalImages=numTrainTotal+numVal; %The final number of images considers adata augmentation only in training dataset

imagesPaths=cell(numFinalImages,1);
labels=zeros(numFinalImages,numCat);
sets=zeros(numFinalImages,1);
imID=zeros(numFinalImages,1);
%Now let's do the organization
idxCont=1;
%Train includes data augmentation
for f=1:numTrain
    idxIm=idxTrain(f);      
    %Reading label
    label=idxStruct(idxIm).label;
    %Labels between 0 and 2
    
    %Reading structures
    annFile=idxStruct(idxIm).annFile;
    try
        lstructures=load([origDatasetDir '/annotations/' annFile],'-ascii');
    catch
        lstructures=zeros(1,numCat-1);
    end
    auxLabel=[label lstructures];
        
    for v=1:numVariations
        %Update the labels
        labels(idxCont,:)=auxLabel;
        imgPath=[imDBFolder '/' num2str(idxIm) '/' num2str(v) '.mat'];
        %Set the paths
        imagesPaths{idxCont}=imgPath;
        sets(idxCont)=1;
        imID(idxCont)=idxIm;
        idxCont=idxCont+1;
    end
end
%Validation without data augmentation
for f=1:numVal
    idxCont=numTrainTotal+f;
    idxIm=idxVal(f);
    sets(idxCont)=2;
    %Reading label
    annFile=idxStruct(idxIm).annFile;
    try
        lstructures=load([origDatasetDir '/annotations/' annFile],'-ascii');
    catch
        lstructures=zeros(1,numCat-1);
    end
    auxLabel=[label lstructures];
    labels(idxCont,:)=auxLabel;
    imID(idxCont)=idxIm;
    %Just the first variation
    imgPath=[imDBFolder '/' num2str(idxIm) '/1.mat'];
    %Set the paths
    imagesPaths{idxCont}=imgPath;
end
imdb.images.paths = imagesPaths ;
imdb.images.data_mean = dataMean;
imdb.images.vdata_mean = vdataMean;
imdb.images.labels = labels ;
imdb.images.imID = imID;
imdb.images.set = sets ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = unique(labels);


function saveImage(imgPath,data,pcoord)

save(imgPath,'data','pcoord');
