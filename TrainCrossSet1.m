function [ Ypredict1 ] = TrainCrossSet1( Xtrain,Ytrain,crossSetLabel )
%NB Train with CV
%Initiate the CV parameter
CV_run=max(crossSetLabel);
CombineSets=cat(2,Xtrain,Ytrain,crossSetLabel);
Ypredict1=zeros(size(Ytrain,1)/CV_run,CV_run);
Ytrain_Index=cat(2,Ytrain,crossSetLabel);
for k=1:CV_run

TrainingSets=CombineSets((CombineSets(1:end,end)~=k),1:end-1);
TestingSets=CombineSets((CombineSets(1:end,end)==k),1:end-1);

Prior1=sum(TrainingSets(:,end)==1)/length(TrainingSets(:,end));
Prior0=sum(TrainingSets(:,end)==0)/length(TrainingSets(:,end));

c1=zeros(1,length(Xtrain));
c0=zeros(1,length(Xtrain));

logic1=(TrainingSets(:,end)==1);
logic0=(TrainingSets(:,end)==0);
c1=logic1'*TrainingSets(:,1:end-1);
c0=logic0'*TrainingSets(:,1:end-1);

CondProb1 = zeros(1,length(Xtrain));
CondProb0 = zeros(1,length(Xtrain));

%smoothing

alpha=1;
CondProb1 =(c1+alpha)/(alpha*length(Xtrain)+sum(c1));
CondProb0 =(c0+alpha)/(alpha*length(Xtrain)+sum(c0));



%Prediction Test set
Ntest=length(TestingSets(:,1));

score_1=zeros(Ntest,1); score_0=zeros(Ntest,1);

for n=1:Ntest
    score_1(n)=sum(log(CondProb1).*TestingSets(n,1:end-1))+log(Prior1);
    score_0(n)=sum(log(CondProb0).*TestingSets(n,1:end-1))+log(Prior0);
    n=n+1;
end


comp=score_1-score_0;


for n=1:Ntest
    if comp(n)>0
        Ypredict1(n,k)=1;
    else
        Ypredict1(n,k)=0;
    end

end

end
for k=1:CV_run
Ypredict_new(find(Ytrain_Index(:,2)==k),1)=Ypredict1(1:end,k);
end
Ypredict1=Ypredict_new;
end

