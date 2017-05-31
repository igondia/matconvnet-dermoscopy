function trainSVMoutput()
inFile='data/melanomas_12/outs2_tr5.mat';
dbFile='data/imdb_fused_2017_tr_1_1_5_test';
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
m_age=zeros(3,1);
s_age=zeros(3,1);
m_area=zeros(3,1);
s_area=zeros(3,1);

for c=1:3
    aux_ages=ages(labels==(c-1) & ages>0);
    aux_area=areas(labels==(c-1));
    aux_gender=gender(labels==(c-1));
    [m_age(c), s_age(c)]=normfit(aux_ages);
    [m_area(c), s_area(c)]=normfit(aux_area);
    p_gender(c,1)=sum(aux_gender==1);
    p_gender(c,2)=sum(aux_gender==2);
    figure(1);subplot(1,3,c);hist(aux_area,0.1:0.2:0.9);
    figure(2);subplot(1,3,c);hist(aux_ages,5:5:85);
end
p_gender=bsxfun(@rdivide,p_gender,sum(p_gender,2));
if(iscell(inFile))
    load(inFile{1});
    fpredictions=zeros(size(outs.predictions));
    for i=1:length(inFile)
        load(inFile{i});
        fpredictions=fpredictions+outs.predictions;
        %fpredictions=max(fpredictions,outs.predictions);%Max
    end
    fpredictions=fpredictions/length(inFile);
    outs.predictions=fpredictions;
else
    load(inFile);
end
addpath('libsvm-gpm/matlab');
for c=1:3
        lAges(:,c) = normpdf(ages,m_age(c),s_age(c));
        lAreas(:,c) = normpdf(areas,m_area(c),s_area(c));
end

lAges=lAges(5:5:end,:);
ages=ages(5:5:end);
lAreas=lAreas(5:5:end,:);
gender=gender(5:5:end,:);


Cs=2.^[-4 -3 -2 -1 0 1 2 3 4 5 6];
for c=1:2
    iAges=lAges(:,c+1)./lAges(:,1);
    iAges(ages<0)=1;
    iAreas=lAreas(:,c+1)./lAreas(:,1);
    iGender=ones(size(iAreas,1),3);
    iGender(gender>0,:)=p_gender(:,gender(gender>0))';
    iGender=iGender(:,c+1)./iGender(:,1);
    
    inputs=double([outs.predictions(:,c) iAges iAreas iGender]);
    normalizer{c}=max(inputs);
    inputs=bsxfun(@rdivide,inputs,normalizer{c});
    labels_aux=outs.labels==c;
    w0=sum(labels_aux)/numel(labels_aux);
    w1=1-w0;
    w0=0.5;
    w1=0.5;
    g=0.1;
    for cost=1:length(Cs)
        opts=sprintf('-s 0 -t 0 -v 5 -w0 %f -w1 %f -c %f -y 1 -g %f -q',w1,w0,Cs(cost),g);
        vals(cost)=svmtrain(double(labels_aux),inputs,opts);
        fprintf('C=%f with AUC=%f\n',Cs(cost),vals(cost));
    end
    [optVal optidx]=max(vals);
    opts=sprintf('-s 0 -t 0 -w0 %f -w1 %f -c %f -y 1 -g %f -q',w0,w1,Cs(optidx),g);
    b(c)=svmtrain(double(labels_aux),inputs,opts);
    for w=1:3
        in=zeros(1,3);
        in(w)=1;
        [~, ~, ws(w)] = svmpredict(double(labels_aux(1)), in, b(c), '-q');
    end
    ws
    [predicted_label, accuracy, outs_clas] = svmpredict(double(labels_aux), inputs, b(c), '-q');
%     weights=zeros(size(labels_aux));
%     weights(labels_aux==0)=w0;
%     weights(labels_aux==1)=w1;
%     b{c} = glmfit(inputs,double(labels_aux),'binomial','weights',weights,'link','logit');
%      b{c}
%     outs_clas = glmval(b{c},inputs,'probit');
    if(sum(labels_aux)>0)
        auc(c)=fAUC(labels_aux>0,outs_clas);
        auc_ant(c)=fAUC(labels_aux>0,outs.predictions(:,c));
    else
        auc(c)=0.5;
        auc_ant(c)=0.5;
    end
%     svmpredict()
end
save('Fusors','b','m_age','m_area','s_age','s_area','p_gender','normalizer');
[mean(auc) auc]
[mean(auc_ant) auc_ant]
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


