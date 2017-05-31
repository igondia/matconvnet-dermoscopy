function trainSVMmulticlassoutput()

load data/idx/fused_2017_aug.mat
dbs=cat(1,ch_cases.db);
idx=find(dbs==1);
ages=cat(1,ch_cases.age);
ages=ages(idx);
labels=cat(1,ch_cases.label);
labels=labels(idx);
gender=cat(1,ch_cases.gender);
gender=gender(idx);
areas=cat(1,ch_cases.area);
areas=areas(idx);
imSize=cat(1,ch_cases.imSize);
imSize=imSize(idx);
load('Hists','hist_area','hist_age','hist_gender','hist_size','vals_area','vals_age','vals_gender','vals_size');

idxArea = knnsearch(vals_area',areas);
idxAge = knnsearch(vals_age',ages);
idxGender = knnsearch(vals_gender',gender);
idxSize = knnsearch(vals_size',imSize);

lAges=zeros(length(idx),3);
lAreas=lAges;
lGender=lAges;
lSize=lAges;


%Likelihoods
for c=1:3
   
    lAges(:,c) = hist_age{c}(idxAge)';
    lAreas(:,c) = hist_area{c}(idxArea)';
    lGender(:,c) = hist_gender{c}(idxGender)';
    lSize(:,c) = hist_size{c}(idxSize)';
end
     
% %Input features
% lAges=bsxfun(@rdivide,lAges,1/length(vals_age));
% lAges(ages<0)=1;
% lAges=lAges(:,2:3);
% 
% lAreas=bsxfun(@rdivide,lAreas,1/length(vals_area));
% lAreas=lAreas(:,2:3);
% 
% lGender=bsxfun(@rdivide,lGender,1/length(vals_gender));
% lGender(gender==0)=1;
% lGender=lGender(:,2:3);
% 
% lSize=bsxfun(@rdivide,lSize,1/length(vals_size));
% lSize=lSize(:,2:3);

%Train the SVM
addpath('libsvm-gpm/matlab');

Cs=2.^[-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6];
Gs=2.^[-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6];
       
inputs=double([lAges lAreas lGender lSize]);
  
[inputs,mu,sigma] = zscore(inputs);
normalizer=[mu; sigma];
labels_aux=labels;
w0=1-sum(labels==0)/numel(labels_aux);
w1=1-sum(labels==1)/numel(labels_aux);
w2=1-sum(labels==2)/numel(labels_aux);
for cost=1:length(Cs)
    for g=1:length(Gs)
        opts=sprintf('-s 0 -t 3 -v 5 -c %f -w0 %f -w1 %f -w2 %f -y 0 -g %f -q',w0,w1,w2,Cs(cost),Gs(g));
        vals(cost,g)=svmtrain(double(labels_aux),inputs,opts);
    end
end
optVal=max(max(vals));
[idxC, idxG]=find(vals==optVal,1,'first');
fprintf('Best Solution with C=%f G=%f and AUC=%f\n',Cs(idxC),Gs(idxG),optVal);
opts=sprintf('-s 0 -t 3 -c %f -g %f -w0 %f -w1 %f -w2 %f -y 0 -b 1 -q',Cs(idxC),Gs(idxG),w0,w1,w2);
b=svmtrain(double(labels_aux),inputs,opts);
[predicted_label, accuracy, outs_clas] = svmpredict(double(labels_aux), inputs, b, '-b 1 -q');

for c=1:2
    if(sum(labels_aux)>0)
        auc(c)=fAUC(labels_aux>0,outs_clas(:,c+1));
    
    else
        auc(c)=0.5;
    end
end


save('Fusors_multiclass','b','normalizer');
[mean(auc) auc]

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


