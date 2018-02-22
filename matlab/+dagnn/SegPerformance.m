classdef SegPerformance < dagnn.Loss
  properties
    metric = 'wseg_bal'
    aug = false;
    labels = [];
    predictions = [];
    imIDs = [];  
    gamma = 1.00;
    Rborder=0.5;
    idxBorders = [1 5];
  end

  properties (Transient)
    perfs=0;
  end

  methods
      function Y = softmax(obj,X)
          E = exp(obj.gamma*bsxfun(@minus, X, max(X,[],2))) ;
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
     
      function compute_wseg_bal(obj,predictions,labels,params)
          predictions=squeeze(predictions);
          pcoordsr = imresize(gather(params{1}),[size(predictions,1) size(predictions,2)],'Method','nearest');
          validos=pcoordsr(:,:,1,:)>=0;
          numValidos=squeeze(sum(sum(validos,1),2));
          %Softmax
          predictions=vl_nnsoftmax(predictions,[],'gamma',obj.gamma).*repmat(validos,[1 1 size(predictions,3) 1]);
          %If we are considering structures ocurring in the lession borders
          if(~isempty(obj.idxBorders))
            validosBorder=pcoordsr(:,:,1,:)>=obj.Rborder;
            numValidosBorder=squeeze(sum(sum(validosBorder,1),2));
            validosBorder=repmat(validosBorder,[1 1 length(obj.idxBorders) 1]);
            predictions(:,:,obj.idxBorders,:)=predictions(:,:,obj.idxBorders,:).*validosBorder;
          end
          %Accumulate at lesion level
          predictions=squeeze(sum(sum(predictions,1),2))';
          predictions1=bsxfun(@rdivide,predictions,numValidos);
          if(~isempty(obj.idxBorders))
              predictions1(:,obj.idxBorders)=bsxfun(@rdivide,predictions(:,obj.idxBorders),numValidosBorder(numValidosBorder>0));
          end
          labels(labels<0)=0;
          %Remove the BG from the analysis
          labels=labels(:,2:end);
          predictions=predictions1(:,2:end);
          
          %Check how many conditions we fulfill
          obj.labels=[obj.labels;gather(labels)];
          obj.predictions=[obj.predictions;predictions];
          numPat=size(obj.labels,2);
          obj.perfs=zeros(1,numPat);
          
          %El de los patrones
          for i=1:1:numPat

              aux_labels=obj.labels(:,i);
              aux_labels(aux_labels>=3)=1;
              aux_pred=obj.predictions(:,i);
              if(sum(aux_labels==3)>0)
                  THneg=0.05;
                  THglobal=0.4;
                  outs=(aux_pred>=THneg & aux_pred<THglobal).*1+(aux_pred>=THglobal).*2;
              else
                  THneg=0.05;
                  THglobal=0.4;
                  outs=(aux_pred>=THneg & aux_pred<THglobal).*1+(aux_pred>=THglobal).*2;
              end
              %Acc2 (comentar si Acc3)
              if(strcmp(obj.metric,'wseg_bal2'))
                outs(outs==2)=1;
                aux_labels(aux_labels==2)=1;
              end
              labelsd=unique(aux_labels);
              nLabels=length(labelsd);
              accuracies=zeros(1,nLabels);
              for l=1:nLabels
                  label=aux_labels==labelsd(l);
                  accuracies(l)=sum(outs==labelsd(l) & label)/sum(label);
              end
              obj.perfs(i) = mean(accuracies);
          end
      end
      
      function compute_mAUC(obj,predictions,labels,params)
          predictions=squeeze(predictions);
          pcoordsr = gpuArray(imresize(gather(params{1}),[size(predictions,1) size(predictions,2)],'Method','nearest'));
          validos=pcoordsr(:,:,1,:)>=0;
          numValidos=squeeze(sum(sum(validos,1),2));
          %Softmax
          predictions=vl_nnsoftmax(predictions,[],'gamma',obj.gamma).*repmat(validos,[1 1 size(predictions,3) 1]);
          %If we are considering structures ocurring in the lession borders
          if(~isempty(obj.idxBorders))
            validosBorder=pcoordsr(:,:,1,:)>=obj.Rborder;
            numValidosBorder=squeeze(sum(sum(validosBorder,1),2));
            validosBorder=repmat(validosBorder,[1 1 length(obj.idxBorders) 1]);
            predictions(:,:,obj.idxBorders,:)=predictions(:,:,obj.idxBorders,:).*validosBorder;
          end
          %Accumulate at lesion level
          predictions=squeeze(sum(sum(predictions,1),2))';
          predictions1=bsxfun(@rdivide,predictions,numValidos);
          if(~isempty(obj.idxBorders))
              predictions1(:,obj.idxBorders)=bsxfun(@rdivide,predictions(:,obj.idxBorders),numValidosBorder(numValidosBorder>0));
          end
          predictions=obj.softmax(predictions1);
          labels(labels<0)=0;
          %Remove the BG from the analysis
          labels=labels(:,2:end);
          predictions=predictions(:,2:end);
          
          
          %Check how many conditions we fulfill
          obj.labels=[obj.labels;labels];
          obj.predictions=[obj.predictions;predictions];
          numPat=size(obj.labels,2);
          obj.perfs=zeros(1,numPat);
          
          %El de los patrones
          for i=1:1:numPat
              labels_aux=obj.labels(:,i);
              if(sum(labels_aux)>0)
                  obj.perfs(i)=obj.fAUC(labels_aux>0,obj.predictions(:,i));
                  
              else
                  obj.perfs(i)=-1;
              end
          end
          
      end
      
      function outputs = forward(obj, inputs, params)
          
          cpreds= gather(inputs{1});
          clabels=gather(inputs{2})';
%           obj.metric='wseg_bal2'; 

          if(strcmp(obj.metric,'wseg_bal2') || strcmp(obj.metric,'wseg_bal3'))
              obj.compute_wseg_bal(cpreds,clabels,params);
          elseif(strcmp(obj.metric,'AUC'))
              obj.compute_mAUC(cpreds,clabels,params);
          end
          outputs{1}=mean(obj.perfs);
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

   function params = initParams(obj)
      % We set it to zero
      params{1} = 0 ;
      
    end

    function obj = SegPerformance(varargin)
      obj.load(varargin) ;
    end
  end
end
