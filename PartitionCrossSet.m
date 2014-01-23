function [ crossSetLabel ] = PartitionCrossSet(Train_size,k )
%initiate label
Initial_Label=(1:k)';
tmp=Train_size/k;
%duplicate the label
label=repmat(Initial_Label,tmp,1);
%shuffling the label
rand('seed',1);
crossSetLabel=label(randperm(size(label,1)),:);


end

