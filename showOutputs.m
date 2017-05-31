function showOutputs(cat)
noCat=8;
warning('off');
load data/idx/fused_2017.mat;
folder='data/melanomas_12';
load([folder 'outs_csv.mat'],'predictions','ids','imids');
predictions=predictions(:,cat);
[predictions idx]=sort(predictions,'descend');
ids=ids(idx);
imids=imids(idx);
for i=1:length(ids)
   
    %Image
    im=imread(ch_cases(imids(i)).impath);
    figure(1);imshow(im);
    predictions(i)
    pause;
end