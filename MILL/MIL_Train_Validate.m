function run = MIL_Train_Validate(data_file, classifier)

global preprocess;
clear run;

% The statistics of dataset
% [X, Y, num_data, num_feature] = Preprocessing(D);
% num_class = length(preprocess.ClassSet);
% class_set = preprocess.ClassSet;
[bags, num_data, num_feature] = MIL_Data_Load(data_file);

% Extract the training and testing data
% X_train = X;
% Y_train = Y;
% Y_test = Y_train;
testindex = 1:num_data;

% Classify with Ensemble 
[test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = MIL_Classify(classifier, bags, bags);      
  
%run.Y_compute = Y_compute; run.Y_prob = Y_prob; run.Y_test = Y_train;
run.bag_label = test_bag_label;
run.inst_label = test_inst_label; 
run.bag_prob = test_bag_prob;
run.inst_prob = test_inst_prob;
% Aggregate the predictions in a shot
 
% Report the performance
run.BagAccu = MIL_Bag_Evaluate(bags(testindex), test_bag_label);
if ~isempty(test_inst_label)
    run.InstAccu = MIL_Inst_Evaluate(bags(testindex), test_inst_label);
end;

run.bag_pred = zeros(length(testindex), 3);
run.bag_pred(:, 1) = (1:length(testindex))';
% run.bag_pred(:, 2) = run.bag_prob; 
%tangÌí¼ÓµÄ´úÂë
% run.bag_pred(:, 2) = cat(1,bags.name);

run.bag_pred(:, 2) = run.bag_label; 
run.bag_pred(:, 3) = [bags(testindex).label]';



if (isfield(preprocess, 'EnforceDistrib') && preprocess.EnforceDistrib == 1)
   for i = 1:num_data, num_pos = num_pos + bags(i).label; end;
   [sort_ret, sort_idx ] = sort(run.bag_pred(:,2));
   threshold = sort_ret(num_data - num_pos + 1);   
   run.bag_pred(:, 3) = (run.bag_pred(:,2) >= threshold);
   run.BagAccu = sum(run.bag_pred(:,3) == run.bag_pred(:,4)) / num_data;
end