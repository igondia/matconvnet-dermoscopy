classdef SegPerformance < dagnn.Loss
  properties
    metric = 'wseg_bal'
    aug = false;
    labels = [];
    predictions = [];
    imIDs = [];
    gamma=20.0;
  end

  properties (Transient)
    perfs=0;
  end

  methods

    function outputs = forward(obj, inputs, params)
        
       predictions= gather(inputs{1});
       labels=gather(inputs{2})';
            
       numCat=size(predictions,3);
       numIm=size(predictions,4);
       predictions=squeeze(gather(inputs{1}));
       pcoordsr = gpuArray(imresize(gather(params{1}),[size(predictions,1) size(predictions,2)],'Method','nearest'));
       validos=pcoordsr(:,:,1,:)>=0;
       predictions=predictions.*repmat(validos,[1 1 size(predictions,3) 1]);
       vals=max(predictions,[],3);
       %We remove non-maximal values
       predictions(bsxfun(@lt,predictions,vals))=0;
       predictions(predictions>0)=1;
       predictions=squeeze(sum(sum(predictions,1),2))';
       predictions=bsxfun(@rdivide,predictions,sum(predictions,2));
       
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
            nn=sum(obj.labels(:,i)==0);
            np1=sum(obj.labels(:,i)==1 | obj.labels(:,i)==3);
            np2=sum(obj.labels(:,i)==2);
            ps=[];
            if(nn>0)
                pn=sum(obj.predictions(obj.labels(:,i)==0,i)<0.05)/nn;
                ps=[ps pn];
            end
            if(np1>0)
                
                pp1=sum(obj.predictions(obj.labels(:,i)==1,i)>=0.05 & obj.predictions(obj.labels(:,i)==1,i)<=0.6 );
                pp1=pp1+sum(obj.predictions(obj.labels(:,i)==3,i)>=0.02 & obj.predictions(obj.labels(:,i)==3,i)<=0.5 );
                pp1=pp1/np1;
                ps=[ps pp1];
            end
            if(np2>0)
                pp2=sum(obj.predictions(obj.labels(:,i)==2,i)>=0.35)/np2;
                ps=[ps pp2];
            end
            obj.perfs(i) =mean(gather(ps));
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
