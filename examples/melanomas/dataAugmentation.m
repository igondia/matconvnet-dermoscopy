%Function that implements the data augmentation process
function [data,pcoord]=dataAugmentation(im,mask,imSize,numAngles,numCrops)

numVariations=numAngles*numCrops;
fprintf('Performing the data augmentation to generate %d variations',numVariations);
data=zeros([imSize(1) imSize(2) 3 numVariations],'single');
pcoord=zeros([imSize(1) imSize(2) 2 numVariations],'single');

%Si la imagen es muy grande, la reducimos
[H W aux]=size(im);
%To avoid operations, if the image is too big, we resize it
if(H>W && H>2048)
    im=imresize(im,[2048 NaN]);
    mask=imresize(mask,[2048 NaN],'nearest');
elseif(W>H && W>2048)
    im=imresize(im,[NaN 2048]);
    mask=imresize(mask,[NaN 2048],'nearest');
end
mask=imdilate(mask,strel('disk',25));
                
%We start by computing different orientations
[imor, maskor]=getOrientedVersions(im,mask,numAngles);
for o=1:size(imor,4)
    fprintf('.');
    imo=imor(:,:,:,o);
    masko=maskor(:,:,o);
    
    mask_aux=rgb2gray(uint8(imo))>0;
    mask_aux=imfill(mask_aux,'holes');
    mask_aux=imclose(mask_aux,strel('disk',10));
    %             figure(2);imshow(imresize(mask_aux,0.25));
    %If we have to crop things
    if(sum(mask_aux(:)==0)>30)
        CC = bwconncomp(mask_aux);
        numPixels = cellfun(@numel,CC.PixelIdxList);
        [~,idx] = max(numPixels);
        mask_aux=zeros(size(mask_aux));
        mask_aux(CC.PixelIdxList{idx}) = 1;
        sf=5;
        [Ho Wo]=size(mask_aux);
        %We do this in small size
        mask_aux=imresize(mask_aux,1/sf,'nearest');
        [~, ~, ~, M] = FindLargestRectangles(mask_aux, [0 0 1], [10 10]);
        if(sum(M(:))>400)
            maskn=imresize(uint8(M),[Ho Wo],'nearest').*masko;
            if(sum(maskn(:))==0)
                masko=maskor(:,:,o);
            else
                masko=maskn;
            end
        else
            masko=maskor(:,:,o);
        end
    end
    %Crop Image using mask
    stats=regionprops(masko,'BoundingBox','Area','Centroid');
    if(length(stats)>1)
        [val idx]=max(cat(1,stats.Area));
    else
        idx=1;
    end
    %Now check the bounding box
    bb=stats(idx).BoundingBox;
    imcropped=imcrop(imo,round(bb));
    maskcropped=imcrop(masko,round(bb));
    
    %             figure(3);imshow(imresize(uint8(imcropped),0.25));;
    %Now we have to obtain the different croppings
    [imcr, pcoordcr]=getCroppedVersions(imcropped,maskcropped,numCrops,imSize);
    data(:,:,:,(o-1)*numCrops+1:o*numCrops)=single(imcr);
    pcoord(:,:,:,(o-1)*numCrops+1:o*numCrops)=single(pcoordcr);
end
fprintf('\n');




%Function that computes the normalized polar coordinates of a mask
function [pcoord]=getPolarCoordinates(mask)

[H W]=size(mask);
stats = regionprops(mask, 'MajorAxisLength','MinorAxisLength','Centroid','Orientation');
alpha = - pi/180 * stats(1).Orientation;
Q = [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)];
x0 = stats(1).Centroid.';
a = stats(1).MajorAxisLength;
b = stats(1).MinorAxisLength;
S = diag([1, a/b]);
C = Q*S*Q';
d = (eye(2) - C)*x0;
d=[0 0]';
Affine=[C d; 0 0 1]';
tform = maketform('affine', Affine);
tform_inv = maketform('affine', inv(Affine));
tpoints=Affine*[1 1 1;1 H 1;W 1 1;W H 1]';
tpoints=tpoints';
    
%Transform the ellipsoidal mask into a normalized circle
maskt= imtransform(mask, tform,'nearest','XYScale',1);
%Generate the polar coordinates
imSize=size(maskt);
stats = regionprops(maskt,'Orientation','Centroid');
center=stats(1).Centroid;
[X,Y] = meshgrid(1:imSize(2),1:imSize(1));
X=X-center(1);
Y=Y-center(2);
R=sqrt(X.^2+Y.^2);
%We set what is outside the mask to negative values
R(maskt==0)=-R(maskt==0);
pcoordt=R;
pcoord= imtransform(pcoordt, tform_inv,'nearest','XYScale',1);
S=size(pcoord);
dif=S(1:2)-size(mask);
dif=ceil(dif/2);
pcoord=pcoord(max(dif(1),1):end-dif(1),max(dif(2),1):end-dif(2));
if(sum(size(pcoord)-size(mask))>0)
    pcoord=imresize(pcoord,size(mask),'Method','nearest');
end
pcoord=pcoord/max(pcoord(:));
%Now computing angles
center=ceil(size(pcoord)/2);
[X,Y] = meshgrid(1:size(pcoord,2),1:size(pcoord,1));
X=X-center(2);
Y=Y-center(1);
%Gives angle between -pi and pi
TH=atan2(-Y,X);
idx=find(TH<0);
TH(idx)=2*pi+TH(idx);
pcoord(:,:,2)=TH;


%Function that gets oriented versions of an image
function [or_versions, or_masks] = getOrientedVersions(im,mask,numAngles)
angle=round(360/(numAngles));
imSize=size(im);
or_versions=single(zeros(imSize(1),imSize(2),3,numAngles));
or_masks=uint8(zeros(imSize(1),imSize(2),numAngles));

for a=1:numAngles
   imr=imrotate(im,angle*(a-1),'nearest','crop');
    maskr=imrotate(mask,angle*(a-1),'nearest','crop');
%     imr=imresize(imr,[imSize(1) imSize(2)]);
%     maskr=imresize(maskr,[imSize(1) imSize(2)]);
    or_versions(:,:,:,a)=imr;
    or_masks(:,:,a)=maskr;
end

function [cr_versions, cr_coords] = getCroppedVersions(im,mask,numCrop,imSize)
[H W]=size(mask);
cr_versions = single(zeros(imSize(1),imSize(2),3,numCrop));
cr_coords = single(zeros(imSize(1),imSize(2),2,numCrop));
if(W>H)
    
    imr=imresize(im,[imSize(1) NaN]);
    maskr=imresize(mask,[imSize(1) NaN]);
    [H W aux]=size(imr);
    dif=W-H;
    step=floor(dif/numCrop);
    for c=1:numCrop
        initp=floor((c-0.5)*step)+1;
        imc=imr(:,initp:initp+imSize(2)-1,:);
        maskc=maskr(:,initp:initp+imSize(2)-1,:);
        pcoordsc=getPolarCoordinates(maskc);
        cr_versions(:,:,:,c)=imc;
        cr_coords(:,:,:,c)=pcoordsc;
    end
else
    
    imr=imresize(im,[NaN imSize(2)]);
    maskr=imresize(mask,[NaN imSize(2)]);
    [H W aux]=size(imr);
    dif=H-W;
    step=floor(dif/numCrop);
    for c=1:numCrop
        initp=floor((c-0.5)*step)+1;
        imc=imr(initp:initp+imSize(1)-1,:,:);
        maskc=maskr(initp:initp+imSize(1)-1,:,:);
        pcoordsc=getPolarCoordinates(maskc);
        cr_versions(:,:,:,c)=imc;
        cr_coords(:,:,:,c)=pcoordsc;
    end    
end
       
