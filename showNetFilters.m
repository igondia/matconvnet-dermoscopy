function showNetFilters(numArch,epoch,channel)
addpath('examples');
dataDir='./data/';
idxDir=[dataDir 'idx/'];
% load([idxDir 'ch_cases.mat']);
run('matlab/vl_setupnn.m') ;
opts.expDir = fullfile('data',['melanomas_' num2str(numArch)]) ;
opts.imdbPath = fullfile('data', 'imdb_.mat');
load([opts.expDir '/net-epoch-' num2str(epoch) '.mat']);
contV=0;
close all;
for l=1:length(net.layers)
    layer=net.layers{l};
    if(strcmp(net.layers{l}.type,'conv'))
        contV=contV+1;
%         figure('units','normalized','outerposition',[0 0 1 1])
        figure(1);
        title(['Layer ' num2str(l)]);
        filters=net.layers{l}.weights{1};
        [H W in out]=size(filters);
        imbyrow=min(20,out);
        rows=ceil(out/imbyrow);
        rows
        imfinal=-1000*ones(rows*H,imbyrow*W);
%         for f=1:out
%             subplot(rows,imbyrow,f);
%             imshow(imresize(filters(:,:,channel,f),[72 72],'nearest'),[]);
        maximo=-10000;
        minimo=10000;
        for r=1:rows
            for c=1:imbyrow
                try
                    nf=(r-1)*imbyrow+c;
                    filter=filters(:,:,channel,nf);
                    filter
%                     maximo=max(abs(filter(:)));
%                     filter=filter/maximo;
%                     minimo=min(filter(:));
%                     maximo=max(filter(:));
%                     filter=(filter-minimo)/(maximo-minimo);
                    imfinal((r-1)*H+1:r*H,(c-1)*W+1:c*W)=filter;
                    maximo=max(maximo,max(filter(:)));
                    minimo=min(minimo,min(filter(:)));
                catch
                    disp(['No filter ' num2str(nf)])
                end
            end
        end
        imfinal=max(imfinal,minimo);
%         maximo=max(abs(imfinal(:)));
%         imfinal=0.5*(imfinal/maximo)+0.5;
        imfinal=imresize(imfinal,10,'nearest');
%         maximo=0.1;
%         minimo=-0.1;
        imfinal=(imfinal-minimo)/(maximo-minimo);
        
        imfinal=im2uint8(imfinal);
        imshow(imfinal,[]);
        imwrite(imfinal,[opts.expDir '/l' num2str(l) 'e' num2str(epoch) '.png'])
%         pause;
    end
end
