function visualizeDataBase

load('data/imdb_64_multilabel.mat');
caso=0;
while 1
    caso=caso+1;
    ha = tight_subplot(6,4,[.01 .03],[.1 .01],[.01 .01])
    for o=1:6
        for s=1:4
            idx=(caso-1)*24+(o-1)*4+s;
            ii=(o-1)*4+s;
%             subplot(6,4,);
            axes(ha(ii));
            image=uint8(imdb.images.data(:,:,:,idx)+imdb.images.data_mean);
            image=imresize(image,4);
            imshow(image)
        end
    end
    pause;
%     close all;
end