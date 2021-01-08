function run = MIL_Train_Test_Simple(bags, trainindex, testindex, classifier)

global preprocess;
clear run;

% The statistics of dataset
%num_class = length(preprocess.ClassSet);
%class_set = preprocess.ClassSet;

% Extract the training and testing data
test_bags = bags(testindex);
train_bags = bags(trainindex);

% Classify with Ensemble 
[test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = MIL_Classify(classifier, train_bags, test_bags);

% run.Y_compute = Y_compute; run.Y_prob = Y_prob; run.Y_test = Y_test;
run.bag_label = test_bag_label; 
run.bag_prob = test_bag_prob; 
run.inst_label = test_inst_label;  
run.inst_prob = test_inst_prob;


% Report the performance
% [run.YY, run.YN, run.NY, run.NN, run.Prec, run.Rec, run.F1, run.Err] = CalculatePerformance(Y_compute, Y_test, class_set);
% if ((preprocess.ComputeMAP == 1) && (length(preprocess.OrgClassSet) == 2)),
%       TrueYprob = Y_prob .* (Y_compute == 1)  + (1 - Y_prob) .* (Y_compute ~= 1);
%       run.AvgPrec = ComputeAP(TrueYprob, Y_test, class_set);
%       run.BaseAvgPrec = ComputeRandAP(Y_test, class_set);       
%       fprintf('AP:%f, Base:%f\n', run.AvgPrec, run.BaseAvgPrec);
% end;    

run.BagAccu = MIL_Bag_Evaluate(test_bags, test_bag_label);
if ~isempty(test_inst_label)
    run.InstAccu = MIL_Inst_Evaluate(test_bags, test_inst_label);
end;