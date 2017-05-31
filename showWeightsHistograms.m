function showWeightsHistograms(modelNum,epochs)

numEpochs=length(epochs);
levels=[];
%Sacamos el numero de niveles convolucionales
load(['data/melanomas_' num2str(modelNum) '/net-epoch-' num2str(epochs(1)) '.mat']);
for l=1:length(net.layers)
    if(strcmp(net.layers{l}.type,'conv'))
        levels=[levels;l];
    end
end
close all;
numLevels=length(levels);
cont=0;
for e=1:numEpochs
    load(['data/melanomas_' num2str(modelNum) '/net-epoch-' num2str(epochs(e)) '.mat']);
    for l=1:numLevels
        cont=cont+1;
%         x=-0.50:0.001:0.50;
        subplot(numEpochs,numLevels,cont);
        maximo=max(max(abs(net.layers{levels(l)}.weights{1}(:))),1e-5);
        [histogram x]=hist(net.layers{levels(l)}.weights{1}(:),linspace(-maximo,maximo,100));
        histogram=histogram/sum(histogram);
        bar(x,histogram);

%         axis([-0.10 0.10 0 0.05]);
        title(['e-' num2str(epochs(e)) ' l-' num2str(levels(l))]);
    end
end

        