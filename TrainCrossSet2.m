function [ Ypredict2 ] = TrainCrossSet2( Xtrain,Ytrain,crossSetLabel )
%LR SGD Train without CV
%Training data with label 0
CV_run=max(crossSetLabel);
lambda=0.001;
alpha=0.078; 
%count=1;
Ypredict2=zeros(size(Ytrain,1)/CV_run,CV_run);
Ytrain_Index=cat(2,Ytrain,crossSetLabel);
CombineSets=cat(2,Xtrain,Ytrain,crossSetLabel);

for k=1:CV_run
%initiate numbers every loop
weights=zeros(size(Xtrain(1,:)));
Obj_Val_Old=0; Obj_Val_New=0; flag=1;

TrainingSets=CombineSets((CombineSets(1:end,end)~=k),1:end-1);
TestingSets=CombineSets((CombineSets(1:end,end)==k),1:end-1);


while flag
%shuffle the order
TrainingSets=TrainingSets(randperm(size(TrainingSets,1)),:);
%count=count+1;
for i=1:length(TrainingSets(:,1))
    p=1/(1+exp(-TrainingSets(i,1:end-1)*weights'));
    weights=weights+alpha*(((TrainingSets(i,end)-p)*TrainingSets(i,1:end-1))-2*lambda*weights);
end

p=1./(1+exp(-TrainingSets(:,1:end-1)*weights'));
Obj_Val_Old=Obj_Val_New;
LCL=sum(TrainingSets(:,end).*log(p)+(1-TrainingSets(:,end)).*log(1-p));
Reg=sum(lambda.*(weights.^2));
Obj_Val_New=LCL-Reg;

if abs(Obj_Val_New-Obj_Val_Old)<0.01
    flag=0;
end

end


for i=1:length(TestingSets(:,1))
    score_1(i)=1/(1+exp(-sum(weights.*TestingSets(i,1:end-1))));
    score_0(i)=1-1/(1+exp(-sum(weights.*TestingSets(i,1:end-1))));
    if score_1(i)>score_0(i)
        Ypredict2(i,k)=1;
    else
        Ypredict2(i,k)=0;
    end
   
end


end

for k=1:CV_run
Ypredict_new(find(Ytrain_Index(:,2)==k),1)=Ypredict2(1:end,k);
end
Ypredict2=Ypredict_new;

end

