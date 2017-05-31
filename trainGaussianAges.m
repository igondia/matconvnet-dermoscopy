function trainGaussianAges()

load data/idx/fused_2017.mat
dbs=cat(1,ch_cases.db);
idx=find(dbs==1);
ages=cat(1,ch_cases.age);
ages=ages(idx);
labels=cat(1,ch_cases.label);
labels=labels(idx);
means=zeros(3,1);
stds=zeros(3,1);
priors=zeros(3,1);
for c=1:3
    aux_ages=ages(labels==(c-1) & ages>0);
    means(c)=mean(aux_ages);
    stds(c)=std(aux_ages);
    priors(c)=sum(labels==(c-1));
end
priors=priors/sum(priors);
gaussians.priors=priors;
gaussians.mu=means;
gaussians.sigma=stds;
gaussians.priors
gaussians.mu
gaussians.sigma

save('gaussianAges.mat','gaussians');