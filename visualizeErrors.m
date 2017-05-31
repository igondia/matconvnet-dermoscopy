function visualizeErrors(inFile,dbFile,class,relevant)

load(inFile);
load(dbFile);
% load(dbFile);
[numCat]=size(outs.predictions,2);
auc=zeros(1,numCat);
for c=1:numCat
    labels_aux=outs.labels==c;
    posClass=1;
    if(sum(labels_aux)>0)
        auc(c)=fAUC(labels_aux>0,outs.predictions(:,c));
    else
        auc(c)=0.5;
    end
end
%Relevant at the end
if(relevant)
    [rankedpred idx]=sort(outs.predictions(:,class),'ascend');
    positives=find(outs.labels(idx)==class);
    rpos=length(idx):-1:1;
    idx=idx(positives);
    rpos=rpos(positives);
    vals=outs.predictions(idx,class);
    for i=1:length(idx)
        fprintf('Relevant image %s at position %d with score %f\n',imdb.images.id{idx(i)},rpos(i),vals(i));
        for v=1:42
            nIm=(idx(i)-1)*imdb.images.numVariations+v;
            load(imdb.images.paths{nIm});
            im=data;%bsxfun(@plus,data,imdb.images.vdata_mean);
            im=im2double(im);
            mask=pcoord(:,:,1)>0;
            displayImage(im,mask);
            pause;
        end
    end
%Non-relevant at the top    
else
    [rankedpred idx]=sort(outs.predictions(:,class),'descend');
    negatives=find(outs.labels(idx)~=class);
    rneg=1:1:length(idx);
    idx=idx(negatives);
    rneg=rneg(negatives);
    vals=outs.predictions(idx,class);
    for i=1:length(idx)
        fprintf('Non-Relevant image %s at position %d with score %f\n',imdb.images.id{idx(i)},rneg(i),vals(i));
        for v=1:42
            nIm=(idx(i)-1)*imdb.images.numVariations+v;
            load(imdb.images.paths{nIm});
            im=data;%bsxfun(@plus,data,imdb.images.vdata_mean);
            im=im2double(im);
            mask=pcoord(:,:,1)>0;
            displayImage(im,mask);
            pause;
        end
    end
end
end

function [auc,fpr,tpr] = fAUC(labels,scores)
    
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
    
    
    auc = sum(tpr.*[(diff(fp)==1); zeros(1,m)])./num_neg;
end

% -------------------------------------------------------------------------
function displayImage(im, mask)
% -------------------------------------------------------------------------
subplot(1,2,1) ;
im=im/255;
image(uint8(255*im));
axis image ;
title('source image') ;

subplot(1,2,2) ;
bmask=(imdilate(mask,strel('disk',3))-mask)>0;
r=im(:,:,1);
g=im(:,:,2);
b=im(:,:,3);
r(bmask)=0;
g(bmask)=1;
b(bmask)=0;
g(mask)=min(g(mask)+30,255);
gtim=255*cat(3,r,g,b);
image(uint8(gtim)) ;
axis image ;
title('ground truth')

cmap = labelColors() ;
end

% -------------------------------------------------------------------------
function cmap = labelColors()
% -------------------------------------------------------------------------
N=21;
cmap = zeros(N,3);
for i=1:N
  id = i-1; r=0;g=0;b=0;
  for j=0:7
    r = bitor(r, bitshift(bitget(id,1),7 - j));
    g = bitor(g, bitshift(bitget(id,2),7 - j));
    b = bitor(b, bitshift(bitget(id,3),7 - j));
    id = bitshift(id,-3);
  end
  cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
end
