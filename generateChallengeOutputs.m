function generateChallengeOutputs(inFile,dbFile,useLabels,method)
% method='avg';
load('data/idx/fused_2017_aug.mat');
if(iscell(inFile) && length(inFile)>1)
    load(inFile{1});
    if(strcmp(method,'avg') || strcmp(method,'normavg') || strcmp(method,'wavg'))
        fpredictions=zeros(size(outs.predictions));
    elseif(strcmp(method,'prod') || strcmp(method,'wprod'))
        fpredictions=ones(size(outs.predictions));
    else
        fpredictions=-1000*ones(size(outs.predictions));
    end
    for i=1:length(inFile)
        load(inFile{i});
        outs.predictions=double(outs.predictions);
        switch method
            case 'avg'
                fpredictions=fpredictions+outs.predictions;
            case 'normavg'
%                 meano=mean(outs.predictions(:));
%                 stdo=std(outs.predictions(:));
%                 [meano stdo]
%                 outs.predictions=bsxfun(@minus,outs.predictions,meano);
                maximos=max(abs(outs.predictions(:)));
                outs.predictions=bsxfun(@rdivide,outs.predictions,maximos);
                fpredictions=fpredictions+outs.predictions;
            case 'prod'    
                maximos=max(outs.predictions(:));
                outs.predictions=bsxfun(@rdivide,outs.predictions,maximos);
                fpredictions=fpredictions.*softmax(outs.predictions);
             case 'wprod'    
                maximos=max(abs(outs.predictions(:)));
                outs.predictions=bsxfun(@rdivide,outs.predictions,maximos);
                preds=softmax(outs.predictions);
                weights=max(exp(outs.predictions),[],2);
                fpredictions=fpredictions.*preds.^repmat(weights,1,3);    
            case 'wavg'
                maximos=max(abs(outs.predictions(:)));
                outs.predictions=bsxfun(@rdivide,outs.predictions,maximos);
                dev=std(outs.predictions,0,2);
                fpredictions=fpredictions+outs.predictions;
                
            case 'max'    
                fpredictions=max(fpredictions,outs.predictions);%Max
                
            case 'imax'
                outs.predictions=softmax(outs.predictions);
                change=max(outs.predictions,[],2)>max(fpredictions,[],2);
                fpredictions(change,:)=outs.predictions(change,:);
            otherwise
                error('not known method');
                
        end
    end
%     if(strcmp(method,'avg') || strcmp(method,'normavg') || strcmp(method,'wavg'))
%          fpredictions=fpredictions/length(inFile);
%     end
    outs.predictions=fpredictions;
else
    if(iscell(inFile))
        inFile=inFile{1};
    end
        
    load(inFile);
end
%Soft-Max
if(sum(outs.predictions(:)<0)>1)
    outs.predictions=softmax(outs.predictions);
end
if(size(outs.predictions,2)==3)
    outs.predictions=outs.predictions(:,2:3);
end
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
[mean(auc) auc]


%Ahora haciendo modificaciones
if(useLabels==1)
    ids=char(imdb.images.id);
    ids=str2num(ids(:,6:end));
    idx=ids<12090;
    outs.predictions(idx,2)=outs.predictions(idx,2)*1e-3;
    idx=ids>=1152&ids<9868;
    outs.predictions(idx,:)=outs.predictions(idx,:)*1e-3;
    idx=ids>=12090;
    outs.predictions(idx,:)=outs.predictions(idx,:)*1.2;
    
    outs.predictions=bsxfun(@rdivide,outs.predictions,max(outs.predictions));
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
    [mean(auc) auc]
 %Use Age
elseif(useLabels==2)
     ids=outs.imIDs;
     ages=cat(1,ch_cases(outs.imIDs).age);
     areas=cat(1,ch_cases(outs.imIDs).area);
     gender=cat(1,ch_cases(outs.imIDs).gender);
     imSize=cat(1,ch_cases(outs.imIDs).imSize);
%      load('gaussianAges');
     load('Fusors','b','m_age','m_area','s_age','s_area','p_gender','normalizer');
     numData=length(outs.labels);
%      ages=
     lAges=zeros(size(outs.predictions,1),size(outs.predictions,2)+1);
     lAreas=lAges;
     lGender=lAges;
     for c=1:3
%         likelihoods(:,c) = normpdf(ages,gaussians.mu(c),gaussians.sigma(c));
        lAges(:,c) = normpdf(ages,m_age(c),s_age(c));
        lAreas(:,c) = normpdf(areas,m_area(c),s_area(c));
     end
     
     lAreas=bsxfun(@rdivide,lAreas,lAreas(:,1));
     lAges=bsxfun(@rdivide,lAges,lAges(:,1));
     lAges=lAges(:,2:3).^0.5;
     lAges(ages<0)=1;
     lAreas=lAreas(:,2:3).^0.0;
     lGender(gender>0,:)=p_gender(:,gender(gender>0))';
     lGender(gender==0,:)=1;
     lGender=bsxfun(@rdivide,lGender,lGender(:,1));
     lGender=lGender(:,2:3).^0.0;
     
     outs.predictions=outs.predictions.*lAges.*lAreas.*lGender;
     minSize=2500;
     outs.predictions(imSize<minSize,2)=outs.predictions(imSize<minSize,2)*0.1;
     outs.predictions=bsxfun(@rdivide,outs.predictions,max(outs.predictions));
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
     [mean(auc) auc]
elseif(useLabels==3)
    ids=outs.imIDs;
     ages=cat(1,ch_cases(outs.imIDs).age);
     areas=cat(1,ch_cases(outs.imIDs).area);
     gender=cat(1,ch_cases(outs.imIDs).gender);
     imSize=cat(1,ch_cases(outs.imIDs).imSize);
     
     load('Hists','hist_area','hist_age','hist_gender','hist_size','vals_area','vals_age','vals_gender','vals_size');
     load('Fusors','b','normalizer');
     idxArea = knnsearch(vals_area',areas);
     idxAge = knnsearch(vals_age',ages);
     idxGender = knnsearch(vals_gender',gender);
     idxSize = knnsearch(vals_size',imSize);
     
     lAges=zeros(size(outs.predictions,1),size(outs.predictions,2)+1);
     lAreas=lAges;
     lGender=lAges;
     lSize=lAges;
     for c=1:3
        lAges(:,c) = hist_age{c}(idxAge)';
        lAreas(:,c) = hist_area{c}(idxArea)';
        lGender(:,c) = hist_gender{c}(idxGender)';
        lSize(:,c) = hist_size{c}(idxSize)';
     end
    
     %Input features
     lAges=bsxfun(@rdivide,lAges,1/length(vals_age));
     lAges(ages<0)=1;
     lAges=lAges(:,2:3);
     
     lAreas=bsxfun(@rdivide,lAreas,1/length(vals_area));
     lAreas=lAreas(:,2:3);
     
     lGender=bsxfun(@rdivide,lGender,1/length(vals_gender));
     lGender(gender==0)=1;
     lGender=lGender(:,2:3);
     
     lSize=bsxfun(@rdivide,lSize,1/length(vals_size));
     lSize=lSize(:,2:3);

    numCat=2;
    auc=zeros(1,numCat);
    exponents=[0.1 0.1]*length(inFile);%[0.4 0.4]; %exponential
%     exponents=[0.25 0.25]; %linear combination
    for c=1:numCat
         inputs=double([lAges(:,c) lAreas(:,c) lGender(:,c) lSize(:,c)]);
%          inputs=bsxfun(@rdivide,inputs,normalizer{c});
         inputs=bsxfun(@minus,inputs,normalizer{c}(1,:));
         inputs=bsxfun(@rdivide,inputs,normalizer{c}(2,:));
         addpath('libsvm-gpm/matlab');
%         outs_clas = glmval(b{c},inputs,'probit');
         [predicted_label, accuracy, outs_clas] = svmpredict(double(labels_aux), inputs, b(c), '-b 1 -q');
         outs_clas=outs_clas(:,2).^exponents(c);
         outs_clas=outs_clas/max(outs_clas);
         outs.predictions(:,c)=outs.predictions(:,c).*outs_clas;
         
    end
    for c=1:numCat 
        labels_aux=outs.labels==c;
        if(sum(labels_aux)>0)
            auc(c)=fAUC(labels_aux>0,outs.predictions(:,c) );
        else
            auc(c)=0.5;
        end
    end
     [mean(auc) auc]
elseif(useLabels==4)
      ids=outs.imIDs;
     ages=cat(1,ch_cases(outs.imIDs).age);
     areas=cat(1,ch_cases(outs.imIDs).area);
     gender=cat(1,ch_cases(outs.imIDs).gender);
     imSize=cat(1,ch_cases(outs.imIDs).imSize);
     
     load('Hists','hist_area','hist_age','hist_gender','hist_size','vals_area','vals_age','vals_gender','vals_size');
     load('Fusors_multiclass','b','normalizer');
     idxArea = knnsearch(vals_area',areas);
     idxAge = knnsearch(vals_age',ages);
     idxGender = knnsearch(vals_gender',gender);
     idxSize = knnsearch(vals_size',imSize);
     
     lAges=zeros(size(outs.predictions,1),size(outs.predictions,2)+1);
     lAreas=lAges;
     lGender=lAges;
     lSize=lAges;
     for c=1:3
        lAges(:,c) = hist_age{c}(idxAge)';
        lAreas(:,c) = hist_area{c}(idxArea)';
        lGender(:,c) = hist_gender{c}(idxGender)';
        lSize(:,c) = hist_size{c}(idxSize)';
     end
    
    
    auc=zeros(1,numCat);
    exponents=[0.5 0.5]; %exponential
    inputs=double([lAges lAreas lGender lSize]);
    inputs=bsxfun(@minus,inputs,normalizer(1,:));
    inputs=bsxfun(@rdivide,inputs,normalizer(2,:));
    addpath('libsvm-gpm/matlab');
    
    [predicted_label, accuracy, outs_svm] = svmpredict(double(labels_aux), inputs, b, '-b 1 -q');
    
    for c=1:numCat 
         outs_clas=outs_svm(:,c+1).^exponents(c);
         outs_clas=outs_clas/max(outs_clas);
         outs.predictions(:,c)=outs.predictions(:,c).*outs_clas;
    end
    for c=1:numCat 
        labels_aux=outs.labels==c;
        if(sum(labels_aux)>0)
            auc(c)=fAUC(labels_aux>0,outs.predictions(:,c) );
        else
            auc(c)=0.5;
        end
    end
     [mean(auc) auc]
end

if(iscell(inFile))
    [folder fileName ext]=fileparts(inFile{1});
else
    [folder fileName ext]=fileparts(inFile);
end
outs.predictions=bsxfun(@rdivide,outs.predictions,max(outs.predictions));
%outs.predictions=outs.predictions/max(outs.predictions)
% probs=[0.20 0.1333];
% for c=1:numCat
%     pred=outs.predictions(:,c);
%     L=length(pred);
%     Ldesired=round(probs(c)*L);
%     svals=sort(pred,'descend'); 
%     prop=svals(Ldesired);
%     outs.predictions(:,c)=1./(1+exp(-a*(pred-prop)));
%     [sum(outs.predictions(:,c)>0.5) Ldesired]
% end
fid=fopen([folder '/out.csv' ],'w');
for i=1:length(outs.labels)
    fprintf(fid,'%s,%.10f,%.10f\n',imdb.images.id{i},outs.predictions(i,1),outs.predictions(i,2));
end
fclose(fid);
predictions=outs.predictions;
ids=imdb.images.id;
imids=outs.imIDs;
save([folder '/outs_csv.mat'],'predictions','ids','imids');

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

function Y= softmax(X)
E = exp(bsxfun(@minus, X, max(X,[],2))) ;
L = sum(E,2) ;
Y = bsxfun(@rdivide, E, L) ;
end
