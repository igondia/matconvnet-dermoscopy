function trainHists()

inFile='data/melanomas_12/outs2_tr5.mat';
dbFile='data/imdb_fused_2017_tr_1_1_5_test';
load data/idx/fused_2017_aug.mat
dbs=cat(1,ch_cases.db);
idx=find(dbs==1);
ages=cat(1,ch_cases.age);
ages=ages(idx);
labels=cat(1,ch_cases.label);
labels=labels(idx);
gender=cat(1,ch_cases.gender);
gender=gender(idx);
areas=cat(1,ch_cases.area);
areas=areas(idx);
imSizes=cat(1,ch_cases.imSize);
imSizes=imSizes(idx);


%vals_area=0.1:0.2:0.9;
vals_area=0.125:0.25:1;
vals_age=[-1 5:5:85];
vals_gender=[0 1 2];
vals_size=[1000 2000 3000 4000 6000];%[1000 2000 2592 3000 4000 6000];


for c=1:3
    aux_age=ages(labels==(c-1) & ages>0);
    aux_area=areas(labels==(c-1));
    aux_gender=gender(labels==(c-1));
    aux_size=imSizes(labels==(c-1));
    hist_area{c} =hist(aux_area,vals_area);
    N=3;
    hist_area{c} = imfilter(hist_area{c},(1/N)*ones(1,N),'replicate');
    hist_area{c}=hist_area{c}/sum(hist_area{c});
    hist_age{c} =hist(aux_age,vals_age);
    hist_age{c} = [hist_age{c}(1) imfilter(hist_age{c}(2:end),(1/N)*ones(1,N),'replicate')];
    hist_age{c}=hist_age{c}/sum(hist_age{c});
    hist_gender{c} =hist(aux_gender,vals_gender);
    hist_gender{c}=hist_gender{c}/sum(hist_gender{c});
    hist_size{c} =hist(aux_size,vals_size);
    hist_size{c}=hist_size{c}/sum(hist_size{c});
    figure(1);subplot(1,3,c);bar(vals_area,hist_area{c});
    figure(2);subplot(1,3,c);bar(vals_age,hist_age{c});
    figure(3);subplot(1,3,c);bar(vals_gender,hist_gender{c});
    figure(4);subplot(1,3,c);bar(vals_size,hist_size{c});
end
figure(1);title('Area');
figure(2);title('Age');
figure(3);title('Gender');
figure(4);title('Size');
save('Hists','hist_area','hist_age','hist_gender','hist_size','vals_area','vals_age','vals_gender','vals_size');


% inFile='data/melanomas_12/outs2_tr5.mat';
% dbFile='data/imdb_fused_2017_tr_1_1_5_test';
% load data/idx/fused_2017_aug.mat
% dbs=cat(1,ch_cases.db);
% idx=find(dbs>5);
% ages=cat(1,ch_cases.age);
% ages=ages(idx);
% labels=cat(1,ch_cases.label);
% labels=labels(idx);
% gender=cat(1,ch_cases.gender);
% gender=gender(idx);
% areas=cat(1,ch_cases.area);
% areas=areas(idx);
% imSizes=cat(1,ch_cases.imSize);
% imSizes=imSizes(idx);
% 
% 
% %vals_area=0.1:0.2:0.9;
% vals_area=0.125:0.25:1;
% vals_age=5:5:85;
% vals_gender=[1 2];
% vals_size=[1000 2000 3000 4000 6000];
% 
% c=1;
%     aux_age=ages(ages>0);
%     aux_area=areas;
%     aux_gender=gender;
%     aux_size=imSizes;
%     hist_area{c} =hist(aux_area,vals_area);
%     N=3;
%     hist_area{c} = imfilter(hist_area{c},(1/N)*ones(1,N),'replicate');
%     hist_area{c}=hist_area{c}/sum(hist_area{c});
%     hist_age{c} =hist(aux_age,vals_age);
%     hist_age{c} = imfilter(hist_age{c},(1/N)*ones(1,N),'replicate');
%     hist_age{c}=hist_age{c}/sum(hist_age{c});
%     hist_gender{c} =hist(aux_gender,vals_gender);
%     hist_gender{c}=hist_gender{c}/sum(hist_gender{c});
%     hist_size{c} =hist(aux_size,vals_size);
%     hist_size{c}=hist_size{c}/sum(hist_size{c});
%     figure(5);subplot(1,3,c);bar(vals_area,hist_area{c});
%     figure(6);subplot(1,3,c);bar(vals_age,hist_age{c});
%     figure(7);subplot(1,3,c);bar(vals_gender,hist_gender{c});
%     figure(8);subplot(1,3,c);bar(vals_size,hist_size{c});
% 
% 
% figure(5);title('Area');
% figure(6);title('Age');
% figure(7);title('Gender');
% figure(8);title('Size');
