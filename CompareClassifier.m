function [ accuracy1, accuracy2, p_value ] = CompareClassifier( Ypredict1, Ypredict2, Ytest, crossSetLabel, isTwoTail )
%compare NB and SGD LR
n=length(Ytest(:,1));
accuracy1=1-sum(abs(Ypredict1-Ytest))/n;
accuracy2=1-sum(abs(Ypredict2-Ytest))/n;
label_no=max(crossSetLabel);
item_no=n/label_no;


%re-parse the Ypredict data with indexes
Ypredict1=cat(2,Ypredict1,crossSetLabel);
Ypredict2=cat(2,Ypredict2,crossSetLabel);
Ytest=cat(2,Ytest,crossSetLabel);

for i=1:label_no
error_1(i)=sum(abs(Ypredict1(find(Ypredict1(:,2)==i),1)-Ytest(find(Ytest(:,2)==i),1)))/item_no;
error_2(i)=sum(abs(Ypredict2(find(Ypredict1(:,2)==i),1)-Ytest(find(Ytest(:,2)==i),1)))/item_no;
Y(i)=error_1(i)-error_2(i);
end
Y_avg=mean(Y);
Y_std=std(Y);
t=Y_avg*sqrt(label_no)/Y_std;

if isTwoTail==0
p_value=tcdf(t,label_no-1);
else
p_value=[tcdf(t,label_no-1),tcdf(-t,label_no-1)];
end



end

