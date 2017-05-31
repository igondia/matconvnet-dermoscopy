function [ap prec]=computeAP(outs,gt)

poslabel=1;
neglabel=0;

[~, si] = sort(-outs);
tp = gt(si)==poslabel;
fp = gt(si)==neglabel;

fp  = cumsum(fp);
tp  = cumsum(tp);
rec = tp/sum(gt>0);
prec= tp./(fp+tp);

% % compute average precision
% % according to before VOC 2010
% ap = 0;
% for t = 0:0.1:1
%     p=max(prec(rec>=t));
%     if isempty(p)
%         p = 0;
%     end
%     ap = ap+p/11;
% end

% compute AP from VOC2010 to ...
ap=VOCap(rec,prec);



function ap = VOCap(rec,prec)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
