function [testInstanceLabel] = PartitionHeldOut (Train_size, k);

%partition random data into k-1 and 1 sets
tmp=Train_size/k;
%initiate the label
label=cat(1,ones(tmp,1),zeros(Train_size-tmp,1));
%shuffling the label	
rand('seed',1);
testInstanceLabel=label(randperm(size(label,1)),:);


end
