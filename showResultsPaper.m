function showResultsPaper()
noCat=8;
categories=cell(noCat,1);
categories{1}='globular/cobblestone';
categories{2}='homogeneous';
categories{3}='reticular';
categories{4}='streaks';
categories{5}='blueveil';
categories{6}='vascularStr';
categories{7}='regressionStr';
categories{8}='unspecific';

colors(1,:)=[0 0 0];
colors(2,:)=[128 0 0];
colors(3,:)=[0 128 0];
colors(4,:)=[128 128 0];
colors(5,:)=[0 0 128];
colors(6,:)=[128 0 128];
colors(7,:)=[0 128 128];
colors(8,:)=[128 128 128];
colors(9,:)=[128 64 128];%[64 0 0];
colors(10,:)=[192 64 0];%[192 0 0];
colors(11,:)=[192 128 128];
colors(12,:)=[192 128 0];
colors(13,:)=[64 0 128];
colors(14,:)=[192 0 128];
colors(15,:)=[64 128 128];
colors(16,:)=[64 128 0];
colors(17,:)=[0 64 0];
colors(18,:)=[128 64 0];
colors(19,:)=[0 192 0];
colors(20,:)=[128 64 128];
colors(21,:)=[0 192 128];
colors(22,:)=[128 192 128];
colors(23,:)=[64 64 0];
colors(24,:)=[192 64 0];
% colors=colors/255;

load data/idx/fused_2017.mat;
vars=randperm(24);
numVar=length(vars);
figure(1);
contSalvar=1;
for i=randperm(2000)
    
    %Image
    im=imread(ch_cases(i).impath);
    subplot(2,4,1);imshow(im);
    %Mask
    try
    mask=imread(ch_cases(i).maskpath);
    catch
        load(ch_cases(i).maskpath);
    end
    subplot(2,4,2);imshow(mask);
    TH=0.05;
    contVars=1;
    %Variations
    for v=1:numVar
        try
            var=vars(v);
            varPath=sprintf('data/db_images/%d/%d.mat',i,var);
            aux=load(varPath);
            varIm{contVars}=uint8(aux.data);
            aux.pcoord(:,:,2)=aux.pcoord(:,:,2)/(2*pi);
            aux.pcoord(:,:,2)=aux.pcoord(:,:,2).*(aux.pcoord(:,:,1)>0);
            pcoordsIm{contVars}=aux.pcoord;
            segPath=sprintf('data/db_segs/%d/%d.mat',i,var);
            load(segPath);
            seg=gather(wmod);
            %Reticular
            seg(:,:,3)=seg(:,:,3)*0.6;
            %Unspecific
            seg(:,:,8)=seg(:,:,8)*0.5;
            %Streaks
            seg(:,:,4)=seg(:,:,4)*5;
            %Remove non-maximal values
            vals=max(seg,[],3);
            %We remove non-maximal values
            seg(bsxfun(@lt,seg,vals))=0;
            totals=squeeze(sum(sum(seg,2),1));
            numP=sum(totals);
            remove=find(totals<TH*numP);
            %         remove=[remove; 1];
            seg(:,:,remove)=0;
            labelsF=find(totals>=TH*numP);
            
            
            
            [vals seg]=max(seg,[],3);
            vmask=aux.pcoord(:,:,1)>0;
            vmask=imerode(vmask,strel('disk',20));
            vmask=imclose(vmask,strel('disk',23));
            vmask=imresize(vmask,size(seg),'method','nearest');
            
            seg=seg.*vmask;
            seg=uint8(seg);
                    maskfill=seg==8;
                    seg2=seg;
                    seg2(maskfill)=0;
                    seg2=imfill(seg2,'holes');
                    seg(maskfill)=seg2(maskfill);
            segShow=ind2rgb(seg,colors);
            segIm{contVars}=uint8(segShow);
            subplot(2,4,2*contVars+1);imshow(varIm{contVars});
            subplot(2,4,2*contVars+2);imshow(segIm{contVars});
            
            
            for l=1:length(labelsF)
                fprintf('%s ',categories{labelsF(l)});
            end
            fprintf('\n');
            contVars=contVars+1;
        catch
        end
        if(contVars==4)
            break;
        end
    end
    if(contVars<4)
        continue;
    end
    disp(['Case ' num2str(i)])
    salvar = input('Save?','s');
    if(strcmp(salvar,'y'))
        fprintf('Salvando\n');
        folder=['resultspaper/' num2str(contSalvar)];
        mkdir(folder);
        imwrite(im,[folder '/im.jpg']);
        imwrite(mask,[folder '/mask.jpg']);
        for v=1:3
            imwrite(varIm{v},[folder '/varim' num2str(v) '.jpg']);
            imwrite(imresize(segIm{v},4,'method','nearest'),[folder '/segim' num2str(v) '.jpg']);
            imwrite(pcoordsIm{v}(:,:,1),[folder '/ratim' num2str(v) '.jpg']);
            imwrite(pcoordsIm{v}(:,:,2),[folder '/angle' num2str(v) '.jpg']);
        end
        contSalvar=contSalvar+1;
    end
    
end
    