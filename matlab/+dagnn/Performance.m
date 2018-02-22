classdef Performance < dagnn.Loss
  properties
    metric = 'auc'
    aug = false;
    labels = [];
    predictions = [];
    imIDs = [];
  end

  properties (Transient)
    perfs=0;
  end

  methods
      
    function Y = softmax(obj,X)
        E = exp(bsxfun(@minus, X, max(X,[],2))) ;
        L = sum(E,2) ;
        Y = bsxfun(@rdivide, E, L) ;
    end
    
     function [auc,fpr,tpr] = fAUC(obj,labels,scores)
         
         if ~islogical(labels)
             error('labels input should be logical');
         end
         if ~isequal(size(labels),size(scores))
             error('labels and scores should have the same size');
         end
         [n,m] = size(labels);
         num_pos = sum(labels);
%          if any(num_pos==0)
%              error('no positive labels entered');
%          end
%          if any(num_pos==n)
%              error('no negative labels entered');
%          end
         
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
     
    function outputs = forward(obj, inputs, params)
        
        %if we just have one output
        if(size(inputs{1},3)==1)
            predictions=squeeze(gather(inputs{1}));
            posClass=1;
            labels=gather(inputs{2})';
        %If we have more than one output
        else
            predictions= gather(inputs{1});
            labels=(gather(inputs{2})-1)';
        end
        if(size(predictions,1)>1 || size(predictions,2)>1)
            predictions=max(max(predictions,[],2),[],1);
        end

        %In case we have 3 inputs is because we have data augmentation in
        %the evaluation 
        if(length(inputs)==3)
            idxIm=unique(inputs{3},'sorted');
            npredictions=zeros(length(idxIm),size(predictions,3),'single');
            nlabels=zeros(length(idxIm),1,'single');
            for i=1:length(idxIm)
                idx=inputs{3}==idxIm(i);

                %Mean 
                auxpred=predictions(:,:,:,idx);
                auxpred=squeeze(auxpred)';                
                auxpred=mean(auxpred);
                npredictions(i,:)=auxpred;

                %Sum
%                 auxpred=predictions(:,:,:,idx);
%                 auxpred=sum(auxpred,4);
%                 auxpred=squeeze(auxpred)';
%                 npredictions(i,:)=auxpred;%(2:3);
                
                %Max en cada channel
%                 auxpred=predictions(:,:,:,idx);
%                 auxpred=max(auxpred,[],4);
% %                 auxpred=squeeze(vl_nnsoftmax(auxpred,[]))';
%                 auxpred=squeeze(auxpred);
%                 npredictions(i,:)=auxpred;%(2:3);

                %Max y los correspondientes
%                 auxpred=predictions(:,:,:,idx);
%                 [val idxb]=max(auxpred(:));
%                 [idxi,idxj] = ind2sub(size(squeeze(auxpred)),idxb);
%                 auxpred=squeeze(vl_nnsoftmax(auxpred(1,1,:,idxj),[]))';
%                 npredictions(i,:)=auxpred(2:3);

                %LSE
%                 auxpred=predictions(:,:,:,idx);
%                 auxpred=squeeze(auxpred)'; 
%                 maximo=max(auxpred(:));
%                 auxpred=log(mean(exp(auxpred-maximo))+1e-100)+maximo;
%                 npredictions(i,:)=auxpred;

                
                %We get the best 5
%                 auxpred=predictions(:,:,:,idx);
%                 auxpred=squeeze(auxpred)'; 
%                 [vals idxbest]=sort(abs(auxpred),'descend');
%                 auxpred=[auxpred(idxbest(1:5,1),1) auxpred(idxbest(1:5,2),2) auxpred(idxbest(1:5,3),3)];
%                 auxpred=mean(auxpred);
%                 npredictions(i,:)=auxpred;%(2:3);

                nlabels(i)=labels(find(idx,1,'first'));
            end
            labels=nlabels;
            predictions=npredictions;
        else
            predictions=squeeze(predictions);
            predictions=predictions';
            idxIm=[];
        end
        obj.labels=[obj.labels;labels];
        obj.predictions=[obj.predictions;predictions];
        obj.imIDs = [obj.imIDs;gather(idxIm)];
        
        if(sum(obj.predictions(:)<0))
            eval_pred=obj.softmax(obj.predictions);
        else
            eval_pred=obj.predictions;
        end
        
        %We quit the background    
        eval_pred=eval_pred(:,2:end);        
        %eval_pred
        switch obj.metric
              case 'auc'
                  [numCat]=size(eval_pred,2);
                  for c=1:numCat
                    labels_aux=obj.labels==c;
                    if(sum(labels_aux)>0)
                        obj.perfs(c)=obj.fAUC(labels_aux>0,eval_pred(:,c));
                    else
                        obj.perfs(c)=-1;
                    end
                  end
            otherwise
                obj.perfs=0;
        end
        outputs{1}=mean(obj.perfs(obj.perfs>=0));
        n = obj.numAveraged ;
        m = n + size(inputs{1},4) ;
        obj.average = gather(outputs{1});
        obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        x=inputs{1};
        if isa(x,'gpuArray')
            derInputs{1} = gpuArray.zeros(size(x),classUnderlying(x)) ;
        else
            derInputs{1} = zeros(size(x),'like',x) ;
        end
    end

    function reset(obj)
      obj.perfs = 0 ;
      obj.average = 0 ;
      obj.labels=[];
      obj.predictions=[];
      obj.imIDs=[];
    end

   
    function obj = Performance(varargin)
      obj.load(varargin) ;
    end
  end
end
