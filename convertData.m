function convertData()

imFolder='data/db_images';
featFolder='data/db_features';
load data/imdb_fused_2017_tr_1_4_0.mat;
[folder fname ext]=fileparts(imdb.images.paths{1});
lFolder=length(folder);
sfolder=regexprep(folder,'db_images','db_features');

% tic
cont=0;
for i=1:length(imdb.images.paths)
    if(rem(i,1000)==0)
        fprintf('%d/%d\n',i,length(imdb.images.paths));
%         toc
%         tic
    end
    fname=imdb.images.paths{i}(lFolder+2:end);
    sFile=[sfolder '/' fname];
    variableInfo = who('-file', sFile);
     % returns true
    if(~ismember('pcoord', variableInfo))
        cont=cont+1;
%         pcoord=aux.pcoord;
%         aux=load(sFile,'data');
%         data=gather(aux.data);
%         saveFile(sFile,data,pcoord);

%         aux=load(imdb.images.paths{i},'pcoord');
%         saveFile(sFile,aux.pcoord);
        
    end
end
cont
function saveFile(sFile,pcoord)
save(sFile,'pcoord','-append');
% save(sFile,'data','pcoord');