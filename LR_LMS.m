function error=LR_LMS( Xtrain, Ytrain, Xtest, Ytest )
%LMS
intercept=ones(size(Xtrain(:,1)));
x_quad=Xtrain.^2;
x_3=Xtrain.^3;
Xtrain_new=cat(2,intercept,Xtrain,x_quad,x_3);
CombineSets=cat(2,Ytrain,Xtrain_new);
weights=ones(size(Xtrain_new(1,:)'));
alpha=1e-5;
n=1;
while n<101
CombineSets=CombineSets(randperm(size(CombineSets,1)),:);
for i=1:length(CombineSets(:,1))
loss=CombineSets(i,1)-CombineSets(i,2:length(CombineSets(1,:)))*weights;
weights=weights+alpha*(loss.*CombineSets(i,2:length(CombineSets(1,:))))';
end
n=n+1;
end

%Construct nonlinear basis for test sets
x_quad_test=Xtest.^2;
x_3_test=Xtest.^3;
intercept_test=ones(size(Xtest(:,1)));
Xtest_new=cat(2,intercept_test,Xtest,x_quad_test,x_3_test);
%Predict Ytest
Ypred_LMS=Xtest_new*weights;
%Compute Error
error=norm(Ypred_LMS-Ytest);
fprintf('Error = %.3f\n', error);

save Ypred_LMS.mat Ypred_LMS;



end

