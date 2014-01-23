function error=LR_NormalEquation( Xtrain, Ytrain, Xtest, Ytest )
%NE
intercept=ones(size(Xtrain(:,1)));
x_quad=Xtrain.^2;
x_3=Xtrain.^3;
Xtrain_new=cat(2,intercept,Xtrain,x_quad,x_3);
weights=inv(Xtrain_new'*Xtrain_new)*Xtrain_new'*Ytrain;


%Construct nonlinear basis for test sets
x_quad_test=Xtest.^2;
x_3_test=Xtest.^3;
intercept_test=ones(size(Xtest(:,1)));
Xtest_new=cat(2,intercept_test,Xtest,x_quad_test,x_3_test);
%Predict Ytest
Ypred_NE=Xtest_new*weights;
%Compute Error
error=norm(Ypred_NE-Ytest);
fprintf('Error = %.3f\n', error);

save Ypred_NE.mat Ypred_NE;



end

