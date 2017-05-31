function cnn_trainMelanomas_diagnosis_mod(numArch)
setenv('LD_LIBRARY_PATH','/usr/local/cuda-7.5/lib64:/usr/lib64/'); 
addpath('examples');
dataDir='./data/';
run('matlab/vl_setupnn.m') ;
%
% opts.expDir = fullfile('data','melanomas-baseline') ;
opts.expDir = fullfile('data',['melanomas_' num2str(numArch)]) ;
opts.imdbPath = fullfile('data', 'imdb_.mat');
opts.train.batchSize = 256 ;
opts.train.numEpochs = 25 ;
opts.train.continue = true ;
opts.train.gpus = [];
opts.train.learningRate=logspace(-1, -4, 20) ;
opts.train.levels_update=-1;
opts.networkType='simplenn';
% opts.train.learningRate = [0.001*ones(1, 25) 0.0001*ones(1, 25)];% 0.00001*ones(1,15)] ;
opts.train.expDir = opts.expDir ;
opts.batchNormalization = false ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightPatterns = 0.5;
opts.train.errorFunction = 'ap' ;
opts.numSet=0;
opts.dbName='fused_2017';

%We just use the challenge to train
opts.trainDB=1;
%We just use the challenge to test
opts.testDB=4;
% opts.train.plotDiagnostics = 1;
% opts = vl_argparse(opts, varargin) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%DESIGN THE NETWORK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(opts.expDir);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%READ THE DATASET%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% netStructure;
if(exist([opts.expDir '/net-init.mat'],'file'))
    disp('Found Init file. Loading');
    load([opts.expDir '/net-init.mat']);
    
    for i=1:numel(net.layers)
        if(isfield(net.layers{i},'momentum'))
            net.layers{i}=rmfield(net.layers{i},'momentum');
        end
    end
    %%Only the options
    [~, net_seg, opts]=netStructure(opts);
else
    [net, net_seg,opts]=netStructure(opts);
end


if(~exist(opts.imdbPath,'file'))
   
    idxDir=[dataDir 'idx/'];
    load([idxDir '' opts.dbName '.mat']);
    
    imdb = getMelanomasImdb(ch_cases,opts.imSize,opts.imdbPath,opts.trainDB,opts.testDB,opts.numSet);
    save(opts.imdbPath,'imdb','-v7.3');
end
load(opts.imdbPath,'imdb');

%If the problem is binary
if(length(imdb.meta.classes)==2)
    posLabel=1;
    negLabel=-1;
    if(min(imdb.images.labels)==0)
        imdb.images.labels(imdb.images.labels==1)=posLabel;
        imdb.images.labels(imdb.images.labels==0)=negLabel;
    else
        imdb.images.labels(imdb.images.labels==1)=negLabel;
        imdb.images.labels(imdb.images.labels==2)=posLabel;
    end
else
    imdb.images.labels=imdb.images.labels+1;
end
imdb.images.instanceWeights=[];
%We remove non-interesting labels
idxUtiles=opts.weightPatterns>0;
imdb.images.idxUtiles=idxUtiles;
imdb.images.labels=imdb.images.labels(:,idxUtiles);
opts.validLabelsError=opts.validLabelsError(idxUtiles);

%If we just learn the diagnostic
if(size(imdb.images.labels,2)==1)
%     labels=imdb.images.labels(:,1);
%     numPos=sum(labels==posLabel);
%     propPos=numPos/length(labels);
%     imdb.images.instanceWeights=(labels==negLabel)*propPos+(labels==posLabel)*(1-propPos);
%     imdb.images.instanceWeights=length(imdb.images.instanceWeights)*imdb.images.instanceWeights/sum(imdb.images.instanceWeights);
    labelsidx=unique(imdb.images.labels);
    props=hist(imdb.images.labels,labelsidx);
    props=props/sum(props);
    numLabels=length(props);
    weights=1./props;
    weights=weights/sum(weights);
    imdb.images.instanceWeights=weights(imdb.images.labels)';
    imdb.images.instanceWeights=imdb.images.instanceWeights/mean(imdb.images.instanceWeights);
    
%    imdb.images.instanceWeights=ones(size(imdb.images.labels));
    %Select just the first label
    imdb.images.labels=imdb.images.labels(:,1);
%We also learn the patterns
else
    nIms=size(imdb.images.labels,1);
    %If the diagnostic is included
    if(opts.weightPatterns(1)>0)
        labels=imdb.images.labels(:,1);
        numPos=sum(labels==posLabel);
        propPos=numPos/length(labels);
        imdb.images.instanceWeights=(labels==negLabel)*propPos+(labels==posLabel)*(1-propPos);
        imdb.images.instanceWeights=length(imdb.images.instanceWeights)*imdb.images.instanceWeights/sum(imdb.images.instanceWeights);
    %If we aim to learn just the patterns    
    else
        imdb.images.instanceWeights=ones(size(imdb.images.labels));
    end
    %Now assign the weights of the patterns with respect to the diagnostic
    opts.weightPatterns=opts.weightPatterns(idxUtiles);
    imdb.images.instanceWeights=imdb.images.instanceWeights.*repmat(opts.weightPatterns,nIms,1);
end
opts.train.validLabelsError=opts.validLabelsError;
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%TRAIN THE NETWORK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Take the mean out and make GPU if neededs
% if opts.train.useGpu
%   imdb.images.data = gpuArray(imdb.images.data) ;
% end
% keyboard;

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train_mod ;
  case 'dagnn', trainFn = @cnn_train_dag;
end
trainidx=find(imdb.images.set == 1);
validx=find(imdb.images.set == 2);
[net, info] = trainFn(net, imdb, @getBatch, opts.train, 'train', trainidx,'val',validx) ;

% --------------------------------------------------------------------
function varargout = getBatch(imdb, batch, networkType,opts)
% --------------------------------------------------------------------
vdata_mean=imdb.images.vdata_mean;
% instanceWeights= imdb.images.instanceWeights(batch,:)';
% instanceWeights=reshape(instanceWeights,1,1,1,length(instanceWeights));
% if(size(instanceWeights,2)>1)
%     labels = imdb.images.labels(batch,:)' ;
% else
%     labels = imdb.images.labels(batch,1)' ;
% end


labels = imdb.images.labels(batch,1)' ;
im=[];
pcoords=[];
numIm=length(batch);
%We read the first one
load(imdb.images.paths{batch(1)},'data','pcoord');
[H W C]=size(data);
im=single(zeros(H,W,C,numIm));
pcoords=single(zeros(H,W,2,numIm));
im(:,:,:,1)=data;
pcoords(:,:,:,1)=pcoord;

%OJO: Para Quitar Polares normalizadas
% center=ceil(size(pcoord)/2);
% [X,Y] = meshgrid(1:size(pcoord,2),1:size(pcoord,1));
% X=X-center(2);
% Y=Y-center(1);
% pcoord(:,:,1)=sqrt(X.^2+Y.^2);
% % pcoord(:,:,1)=pcoord(:,:,1)/max(max(pcoord(:,:,1)));
% pcoord(:,:,1)=pcoord(:,:,1)/center(1);
% pcoord(:,:,1)=min(pcoord(:,:,1),1);
% %Gives angle between -pi and pi
% TH=atan2(-Y,X);
% idx=find(TH<0);
% TH(idx)=2*pi+TH(idx);
% pcoord(:,:,2)=TH;
% pcoords(:,:,:,1)=pcoord;
%OJO: Para Quitar Polares normalizadas

for i=2:numIm
    aux=load(imdb.images.paths{batch(i)},'data','pcoord');
    im(:,:,:,i)=aux.data;
    pcoords(:,:,:,i)=aux.pcoord;
%OJO: Para Quitar Polares normalizadas
%     pcoords(:,:,:,i)=pcoord;
%OJO: Para Quitar Polares normalizadas
end
im=bsxfun(@minus,im,vdata_mean);

switch networkType
    case 'simplenn'
      varargout = {im, pcoords,labels};%,instanceWeights} ;
    case 'dagnn'
      varargout{1} = {'input', im, 'label', labels,'pcoords',pcoords};%,'instanceWeights',instanceWeights} ;
end
  
% --------------------------------------------------------------------
function imdb = getMelanomasImdb(idxStruct,imSize,imDBPath,trainDB,testDB,numSet)
% --------------------------------------------------------------------
imDBFolder='./data/db_images';%imDBPath(1:end-6);
mkdir(imDBFolder);
mkdir([imDBFolder '_icons']);
%parameters of the dataset
numAngles=6;
numCrops=4;
transformImage=0;
numSets=5;

numVariations=numAngles*numCrops;
%For test, we just consider the images in the challenge set (for better comparisons)
dbidx=cat(1,idxStruct(:).db);
%Select the corresponding images if the selected datasets
idxTrain=find(ismember(dbidx,trainDB));
idxTest=find(ismember(dbidx,testDB));
numImagesTrain=length(idxTrain);
numImages=length(dbidx);
%Now, if we have a numSet>0, we need to quit the corresponding images
if(numSet>0 && sum(ismember(testDB,trainDB))>0)
    idxTest=numSet:numSets:length(idxTest);
    idxtrts=find(dbidx(idxTrain)==testDB);
    idxTrain(idxtrts(idxTest))=[];
end


numTrain=length(idxTrain);
numTest=length(idxTest);
numTrainTotal=numVariations*numTrain;
numTotalImages=numVariations*numImages;
numFinalImages=numTrainTotal+numTest;
cont=1;
%numCategories: 1 diagnosis + 9 structural patterns
numCat=10;
data=single(zeros(imSize(1),imSize(2),3));
pcoord=single(zeros(imSize(1),imSize(2),2));

% labels=zeros(numImages,numCat);
strAdd='/'; 

%If we have already computed the images, we can jump to the next step
[result numDoneImages]=unix(['ls -1 ' imDBFolder '/*-*.mat | wc -l']);
%We rest 1 to remove the mean file
numDoneTotImages=str2num(numDoneImages);
if(isempty(numDoneTotImages))
    numDoneTotImages=0;
end
% numDoneTotImages=numTotalImages;
%If it is not done;
if(numDoneTotImages<numTotalImages)
    
    %Generating the whole set of images
    parfor f=1:3602%numImages
        idxIm=f;
        idxCont=1;
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
            %         [folder file ext]=fileparts(idxStruct(idxIm).impath);
            %         folder=regexprep(folder,'_Data','_Data_orig');
            %         im=imread([folder  '' strAdd '' file '' ext]);
            im=imread(idxStruct(idxIm).impath);
            %Passing it to grayscale => CAMBIAR
            im=single(im);
            %Reading mask
            %         [folder file ext]=fileparts(idxStruct(idxIm).maskpath);
            %         folder=regexprep(folder,'_Data','_Data_orig');
            try
                mask=imread(idxStruct(idxIm).maskpath);
            catch
                aux=load(idxStruct(idxIm).maskpath);
                mask=aux.mask;
            end

	    %Si la imagen es muy grande, la reducimos
            [H W aux]=size(im);
	    if(H>W && H>2048)
		im=imresize(im,[2048 NaN]);
		mask=imresize(mask,[2048 NaN],'nearest');
	    elseif(W>H && W>2048)	
		im=imresize(im,[NaN 2048]);
	        mask=imresize(mask,[NaN 2048],'nearest');
	    end
            vmin=min(mask(:));
            if(vmin==255)
                vmin=0;
            end
            mask=mask>vmin;
            mask=imdilate(mask,strel('disk',25));
            

            %We start by computing different orientations
            [imor, maskor]=getOrientedVersions(im,mask,numAngles);
            for o=1:size(imor,4)
                imo=imor(:,:,:,o);
                masko=maskor(:,:,o);
                
%                 if(o==12)
%                     keyboard;
%                 end
                mask_aux=rgb2gray(uint8(imo))>0;
                mask_aux=imfill(mask_aux,'holes');
                mask_aux=imclose(mask_aux,strel('disk',10));
                %             figure(2);imshow(imresize(mask_aux,0.25));
                %If we have to crop things
                if(sum(mask_aux(:)==0)>30)
                    CC = bwconncomp(mask_aux);
                    numPixels = cellfun(@numel,CC.PixelIdxList);
                    [~,idx] = max(numPixels);
                    mask_aux=zeros(size(mask_aux));
                    mask_aux(CC.PixelIdxList{idx}) = 1;
                    sf=5;
                    [Ho Wo]=size(mask_aux);
                    %We do this in small size
                    mask_aux=imresize(mask_aux,1/sf,'nearest');
                    [~, ~, ~, M] = FindLargestRectangles(mask_aux, [0 0 1], [10 10]);
                    if(sum(M(:))>400)
                        maskn=imresize(uint8(M),[Ho Wo],'nearest').*masko;
                        if(sum(maskn(:))==0)
                            masko=maskor(:,:,o);
                        else
                            masko=maskn;
                        end
                    else
                        masko=maskor(:,:,o);
                    end
                end
                
                %             figure(2);imshow(imresize(uint8(imo),0.25));;
                %Crop Image using mask
                stats=regionprops(masko,'BoundingBox','Area','Centroid');
                if(length(stats)>1)
                    [val idx]=max(cat(1,stats.Area));
                else
                    idx=1;
                end
                %Now check the bounding box
                try
                    bb=stats(idx).BoundingBox;
                    imcropped=imcrop(imo,round(bb));
                    maskcropped=imcrop(masko,round(bb));
                    %             figure(3);imshow(imresize(uint8(imcropped),0.25));;
                    %Now we have to obtain the different croppings
                    [imcr, pcoordcr]=getCroppedVersions(imcropped,maskcropped,numCrops,imSize);
                    for c=1:numCrops
                        
                        %Store the results
                        data=single(imcr(:,:,:,c));
                        pcoord=single(pcoordcr(:,:,:,c));
                        imgPath=[imDBFolder '/' num2str(f) '/' num2str(idxCont) '.mat'];
                        imgIPath=[imDBFolder '_icons/' num2str(f) '/' num2str(idxCont) '.jpg'];
                        saveImage(imgPath,data,pcoord);
                        imwrite(imresize(uint8(data),0.10),imgIPath);
                        %                                     figure(4);imshow(uint8(imcr(:,:,:,c)));
                        %                  pause;
                        idxCont=idxCont+1;
                    end

                catch e
                    
                 disp(['Error in image ' num2str(f)])
                end
            end
        end
    end
end

dataMeanFile=[imDBFolder '/dataMeanLabels.mat'];
if(exist(dataMeanFile,'file'))
    load(dataMeanFile, 'dataMean','vdataMean');
else
    dataMean=zeros(imSize(1),imSize(2),3);
    for f=1:2000%numImages
        disp(['Gathering mean from doc ' num2str(f) '/' num2str(numImages)])
        for v=1:numVariations
            
            imgPath=[imDBFolder '/' num2str(f) '/' num2str(v) '.mat'];
            aux=load(imgPath,'data');
            data=aux.data;
            dataMean=dataMean+double(data);
        end
    end
    dataMean=single(dataMean/numTotalImages);
    vdataMean=mean(mean(dataMean,2),1);
    if(sum(isinf(dataMean(:))))
        disp('Overflow in dataMean');
    end
    save([imDBFolder '/dataMeanLabels.mat'], 'dataMean','vdataMean');
    
%     %Re-escale the images
%     parfor f=1:numImages
%         for v=1:numVariations
%             imgPath=[imDBFolder '/' num2str(f) '-' num2str(v) '.mat'];
%             aux=load(imgPath);
%             data=aux.data;
%             pcoord=aux.pcoord;
%             data = data-dataMean;
%             saveImage(imgPath,data,pcoord);
%         end
%     end
end

imagesPaths=cell(numFinalImages,1);
labels=zeros(numFinalImages,numCat);
sets=zeros(numFinalImages,1);
imID=zeros(numFinalImages,1);
%Now let's do the organization
idxCont=1;
for f=1:numTrain
    idxCont=(f-1)*numVariations+1;
    idxIm=idxTrain(f);
          
    %Reading label
    label=idxStruct(idxIm).label;
    %Labels between 0 and 2
    %Reading structures
    annFile=idxStruct(idxIm).annFile;
    try
        lstructures=load(['data/annotations/' annFile],'-ascii');
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
        set(idxCont)=1;
        imID(idxCont)=idxIm;
        idxCont=idxCont+1;
    end
end

for f=1:numTest
    idxCont=numTrainTotal+f;
    idxIm=idxTest(f);

    %Reading label
    label=idxStruct(idxIm).label;
    %Labels between 0 and 2
    %Reading structures
    annFile=idxStruct(idxIm).annFile;
    try
        lstructures=load(['data/annotations/' annFile],'-ascii');
    catch
        lstructures=zeros(1,numCat-1);
    end
    auxLabel=[label lstructures];
    
    set(idxCont)=2;
    %Update the labels
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
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = unique(labels);



%Function that gets oriented versions of an image
function [or_versions, or_masks] = getOrientedVersions(im,mask,numAngles)
angle=round(360/(numAngles));
imSize=size(im);
or_versions=single(zeros(imSize(1),imSize(2),3,numAngles));
or_masks=uint8(zeros(imSize(1),imSize(2),numAngles));

% mask=uint8(mask)+1;

for a=1:numAngles
   imr=imrotate(im,angle*(a-1),'nearest','crop');
    maskr=imrotate(mask,angle*(a-1),'nearest','crop');
%     imr=imresize(imr,[imSize(1) imSize(2)]);
%     maskr=imresize(maskr,[imSize(1) imSize(2)]);
    or_versions(:,:,:,a)=imr;
    or_masks(:,:,a)=maskr;
end

function [cr_versions, cr_coords] = getCroppedVersions(im,mask,numCrop,imSize)
[H W]=size(mask);
% stats = regionprops(mask, 'MajorAxisLength','MinorAxisLength','Centroid','Orientation');
cr_versions = single(zeros(imSize(1),imSize(2),3,numCrop));
% cr_masks=single(zeros(imSize(1),imSize(2),numCrop));
cr_coords = single(zeros(imSize(1),imSize(2),2,numCrop));


%EStimate an ellipsoid with the mask and transform it into a circle
% [pcoords]=getPolarCoordinates(mask);

    
if(W>H)
    
    imr=imresize(im,[imSize(1) NaN]);
    maskr=imresize(mask,[imSize(1) NaN]);
%     pcoordsr=imresize(pcoords,[imSize(1) NaN],'Method','nearest');
    [H W aux]=size(imr);
    dif=W-H;
    step=floor(dif/numCrop);
    for c=1:numCrop
        initp=floor((c-0.5)*step)+1;
        imc=imr(:,initp:initp+imSize(2)-1,:);
        maskc=maskr(:,initp:initp+imSize(2)-1,:);
%         pcoordsc=pcoordsr(:,(c-1)*step+1:(c-1)*step+imSize(1),:);
        pcoordsc=getPolarCoordinates(maskc);
        cr_versions(:,:,:,c)=imc;
        cr_coords(:,:,:,c)=pcoordsc;
%         figure(c);imshow(imc(:,:,1),[]);
%         figure(numCrop+c);imshow(imc(:,:,1).*(pcoordsc(:,:,1)),[]);
%         figure(2*numCrop+c);imshow(imc(:,:,1).*(pcoordsc(:,:,2)),[]);
    end
else
    
    imr=imresize(im,[NaN imSize(2)]);
    maskr=imresize(mask,[NaN imSize(2)]);
%     pcoordsr=imresize(pcoords,[NaN imSize(2)],'Method','nearest');
    [H W aux]=size(imr);
    dif=H-W;
    step=floor(dif/numCrop);
    for c=1:numCrop
        initp=floor((c-0.5)*step)+1;
        imc=imr(initp:initp+imSize(1)-1,:,:);
        maskc=maskr(initp:initp+imSize(1)-1,:,:);
        pcoordsc=getPolarCoordinates(maskc);
%         pcoordsc=pcoordsr((c-1)*step+1:(c-1)*step+imSize(1),:,:);
%         maskc=maskr((c-1)*step+1:(c-1)*step+imSize(1),:);
%         imshow(imc(:,:,1),[])
        cr_versions(:,:,:,c)=imc;
        cr_coords(:,:,:,c)=pcoordsc;
%         cr_masks(:,:,c)=maskc;
%         figure(c);imshow(imc(:,:,1),[])
%         figure(numCrop+c);imshow(imc(:,:,1).*(pcoordsc(:,:,1)),[]);
%         figure(2*numCrop+c);imshow(imc(:,:,1).*(pcoordsc(:,:,2)),[]);
    end
    
end

%Function that gets cropped versions of an image
function imcropped=cropLession(im,mask,imSize)

[H W]=size(mask);
stats = regionprops(mask, 'MajorAxisLength','MinorAxisLength','Centroid','Orientation');
if(H>W)
    imr=imresize(im,[NaN 256]);
    maskr=imresize(mask,[NaN 256]);
    imr=imresize(im,[256 NaN]);
    maskr=imresize(mask,[256 NaN]);
end


%Function that gets cropped versions of an image
function [imt maskt]=transformLession(im,mask)

[H W]=size(mask);
stats = regionprops(mask, 'MajorAxisLength','MinorAxisLength','Centroid','Orientation');
alpha = - pi/180 * stats(1).Orientation;
Q = [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)];
x0 = stats(1).Centroid.';
a = stats(1).MajorAxisLength;
b = stats(1).MinorAxisLength;
S = diag([1, a/b]);
C = Q*S*Q';
d = (eye(2) - C)*x0;
d=[0 0]';
Affine=[C d; 0 0 1]';
tform = maketform('affine', Affine);
tpoints=Affine*[1 1 1;1 H 1;W 1 1;W H 1]';
tpoints=tpoints';
maxpoint=max(tpoints);
minpoint=min(tpoints);
    
%Marco el centro para identificarlo
[imt XData2 YData2]= imtransform(im, tform,'XData',[minpoint(1) maxpoint(1)], 'YData',[minpoint(2) maxpoint(2)]);%,'Size',[Size(2) Size(1)]);%,'XYScale',1);
[maskt XData22 YData22]= imtransform(mask, tform,'XData',[minpoint(1) maxpoint(1)], 'YData',[minpoint(2) maxpoint(2)]);%,'Size',[Size(2) Size(1)]);%,'XYScale',1);

% stats = regionprops(maskcircle,'Orientation','Centroid','MajorAxisLength','MinorAxisLength')
% center=stats(1).Centroid;
% centerWrite=Affine*[x0(1) x0(2) 1]';
% centerWrite=centerWrite(1:2)';
% %     radius=0.5*stats(1).MajorAxisLength;
% %     pdist2([center 1],(Affine*[x0(1) x0(2) 1]')')
% [I,J]=find(maskcircle>0);
% lidx=find(maskcircle>0);
% radios=pdist2([J I],center);
% [radius idx]=max(radios);

%Function that gets cropped versions of an image
function [pcoord]=getPolarCoordinates(mask)

[H W]=size(mask);
stats = regionprops(mask, 'MajorAxisLength','MinorAxisLength','Centroid','Orientation');
alpha = - pi/180 * stats(1).Orientation;
Q = [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)];
x0 = stats(1).Centroid.';
a = stats(1).MajorAxisLength;
b = stats(1).MinorAxisLength;
S = diag([1, a/b]);
C = Q*S*Q';
d = (eye(2) - C)*x0;
d=[0 0]';
Affine=[C d; 0 0 1]';
tform = maketform('affine', Affine);
tform_inv = maketform('affine', inv(Affine));
tpoints=Affine*[1 1 1;1 H 1;W 1 1;W H 1]';
tpoints=tpoints';
maxpoint=max(tpoints);
minpoint=min(tpoints);
    
%Transformo en círculo
[maskt XData2 YData2]= imtransform(mask, tform,'nearest','XYScale',1);
%Genero las coordenadas polares
imSize=size(maskt);
stats = regionprops(maskt,'Orientation','Centroid');
center=stats(1).Centroid;
[X,Y] = meshgrid(1:imSize(2),1:imSize(1));
X=X-center(1);
Y=Y-center(2);
pcoordt=zeros(imSize);
% [TH,R]=cart2pol(X,Y);
R=sqrt(X.^2+Y.^2);
%Ponemos a negativo lo que caiga fuera de la máscara, para usarlo cuando
%sea necesario
R(maskt==0)=-R(maskt==0);
%Normalizamos
% R=R/max(R(:));
pcoordt=R;
[pcoord XData2 YData2]= imtransform(pcoordt, tform_inv,'nearest','XYScale',1);
S=size(pcoord);
dif=S(1:2)-size(mask);
dif=ceil(dif/2);
pcoord=pcoord(max(dif(1),1):end-dif(1),max(dif(2),1):end-dif(2));
if(sum(size(pcoord)-size(mask))>0)
    pcoord=imresize(pcoord,size(mask),'Method','nearest');
end
pcoord=pcoord/max(pcoord(:));
%An dilate masks a little bit (it is better dilating after transformation)
% pcoord=imdilate(pcoord,strel('disk',25));
%Now computing angles
center=ceil(size(pcoord)/2);
[X,Y] = meshgrid(1:size(pcoord,2),1:size(pcoord,1));
X=X-center(2);
Y=Y-center(1);
%Gives angle between -pi and pi
TH=atan2(-Y,X);
idx=find(TH<0);
TH(idx)=2*pi+TH(idx);
pcoord(:,:,2)=TH;
% close all;
% [mask_recovered XData2 YData2]= imtransform(maskt, tform_inv,'nearest');
% mask_recovered1=mask_recovered(dif(1):end-dif(1),dif(2):end-dif(2));
% figure(1);imshow(mask);figure(2);imshow(maskt);figure(3);imshow(mask_recovered);figure(4):imshow(mask_recovered1);figure(5):imshow(pcoord(:,:,1),[]);figure(6);imshow(abs(pcoord(:,:,2)),[])
% pause;

function saveImage(imgPath,data,pcoord)

save(imgPath,'data','pcoord');
