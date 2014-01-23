function error=LR_SteepestDescent( Xtrain, Ytrain, Xtest, Ytest )
%Batch
%Construct nonlinear basis function
intercept=ones(size(Xtrain(:,1)));
x_quad=Xtrain.^2;
x_3=Xtrain.^3;
Xtrain_new=cat(2,intercept,Xtrain,x_quad,x_3);
%Initiate weights
weights=ones(size(Xtrain_new(1,:)'));
%initiate alpha
alpha=1e-5;
for n=1:100
loss=Ytrain-Xtrain_new*weights;
weights=weights+alpha*(loss'*Xtrain_new)';
end
%Construct nonlinear basis for test sets
x_quad_test=Xtest.^2;
x_3_test=Xtest.^3;
intercept_test=ones(size(Xtest(:,1)));
Xtest_new=cat(2,intercept_test,Xtest,x_quad_test,x_3_test);
%Predict Ytest
Ypred_SD=Xtest_new*weights;
%Compute Error
error=norm(Ypred_SD-Ytest);
fprintf('Error = %.3f\n', error);

save Ypred_SD.mat Ypred_SD;


end

