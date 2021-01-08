% Input pararmeter: 
% D: data array, including the feature data and output class
% outputfile: the output file name of classifiers
function run = MIL_Train_Test_Validate(data_file, classifier_wrapper_handle, classifier)

global preprocess;

% [X, Y, num_data, num_feature] = preprocessing(D);
% clear D;
[bags, num_data, num_feature] = MIL_Data_Load(data_file);

if isfield(preprocess, 'test_file') && ~isempty(preprocess.test_file)
    [test_bags, num_test_data, num_feature] = MIL_Data_Load(preprocess.test_file);
    bags = [bags test_bags];
    splitboundary = num_data;
    num_data = num_data + num_test_data;    
else
    if (preprocess.TrainTestSplitBoundary > 0),
        splitboundary = preprocess.TrainTestSplitBoundary;
    else
        splitboundary = fix(num_data / (-preprocess.TrainTestSplitBoundary));
    end;
end;
testindex = splitboundary+1:num_data;
trainindex = 1:splitboundary;

run = feval(classifier_wrapper_handle, bags, trainindex, testindex, classifier);
  
run.bag_pred = zeros(length(testindex), 3);
run.bag_pred(:, 1) = (1:length(testindex))';
run.bag_pred(:, 2) = run.bag_prob; 
run.bag_pred(:, 3) = run.bag_label; 
run.bag_pred(:, 4) = [bags(testindex).label]';

if (isfield(preprocess, 'EnforceDistrib') && preprocess.EnforceDistrib == 1)
   num_pos = 0;
   for i = 1:num_data, num_pos = num_pos + bags(i).label; end;   
   num_pos = round((num_pos / num_data) * length(testindex));   %the expected # of pos bags in the testing data
   
   [sort_ret, sort_idx ] = sort(run.bag_pred(:,2));
   threshold = sort_ret(length(testindex) - num_pos + 1);   
   run.bag_pred(:, 3) = (run.bag_pred(:,2) >= threshold);   
   run.BagAccu = sum(run.bag_pred(:,3) == run.bag_pred(:,4)) / length(testindex);
end