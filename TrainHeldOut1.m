function [ Ypredict1 ] = TrainHeldOut1( Xtrain,Ytrain,testInstanceLabel )
%NB Train without CV
%Training data with label 0

CombineSets=cat(2,Xtrain,Ytrain,testInstanceLabel);
TrainingSets=CombineSets((CombineSets(1:end,end)==0),1:end-1);
TestingSets=CombineSets((CombineSets(1:end,end)==1),1:end-1);


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
Ypredict1=zeros(Ntest,1);

for n=1:Ntest
    if comp(n)>0
        Ypredict1(n)=1;
    else
        Ypredict1(n)=0;
    end
    
end


end

