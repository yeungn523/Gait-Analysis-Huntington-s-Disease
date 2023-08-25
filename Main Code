% data prepare
data_control1 = load('data/control1m.mat');
data_control2 = load('data/control2m.mat');
data_control3 = load('data/control3m.mat');
data_control4 = load('data/control4m.mat');
data_control5 = load('data/control5m.mat');

data_hunt14 = load('data/hunt14m.mat');
data_hunt13 = load('data/hunt13m.mat');
data_hunt1 = load('data/hunt1m.mat');
data_hunt8 = load('data/hunt8m.mat');
data_hunt5 = load('data/hunt5m.mat');

% Combine the data from each .mat file into trainData_control and trainData_hunt
trainData_control = [data_control1.val; data_control2.val; data_control3.val; data_control4.val; data_control5.val];
trainData_hunt = [data_hunt14.val; data_hunt13.val; data_hunt1.val; data_hunt8.val; data_hunt5.val];

Data{1} = data_control1.val;
Data{2} = data_control2.val;
Data{3} = data_control3.val;
Data{4} = data_control4.val;
Data{5} = data_control5.val;

Data{6} = data_hunt14.val;
Data{7} = data_hunt13.val;
Data{8} = data_hunt1.val;
Data{9} = data_hunt8.val;
Data{10} = data_hunt5.val;

%% test data
data_control6 = load('data/control6m.mat');
data_control7 = load('data/control7m.mat');
data_control8 = load('data/control8m.mat');
data_control9 = load('data/control9m.mat');
data_control10 = load('data/control10m.mat');

data_hunt2 = load('data/hunt2m.mat');
data_hunt3 = load('data/hunt3m.mat');
data_hunt4 = load('data/hunt4m.mat');
data_hunt6 = load('data/hunt6m.mat');
data_hunt7 = load('data/hunt7m.mat');

TData{1} = data_control6.val;
TData{2} = data_control7.val;
TData{3} = data_control8.val;
TData{4} = data_control9.val;
TData{5} = data_control10.val;

TData{6} = data_hunt2.val;
TData{7} = data_hunt3.val;
TData{8} = data_hunt4.val;
TData{9} = data_hunt6.val;
TData{10} = data_hunt7.val;

for i=1:10
    Data{i} = mapminmax(fillmissing(Data{i}(2, 1:18000),'linear'), 0, 1)';
end

Q = 2;      % state num
M = 2;      % mix num
p = 2;      % feature dim

% Train Gmm-Hmm model
[p_start, A, phi, loglik] = ChmmGmm(Data, Q, M);
[p_start, A, phi, loglik] = ChmmGmm(Data, Q, M, 'p_start0', p_start0, 'A0', A0, 'phi0', phi0, 'cov_type', 'diag', 'cov_thresh', 1e-1)

% Calculate p(X) & vertibi decode
% majority vote
path_list = 0;
for idx=1:5
     logp_xn_given_zn = Gmm_logp_xn_given_zn(Data{idx}, phi);
     [~,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);
     path = LogViterbiDecode(logp_xn_given_zn, p_start, A);
     if length(find(path==1)) > length(path)/2
         path_list = path_list + 1;
     end
 end
 
 if path_list > 2.5
     control = 1;
     hunt = 2;
 else
     control = 2;
     hunt = 1;
 end
 
 labels = [ones(5, 1)*control; ones(5, 1)* hunt];
 predict_labels = [zeros(10, 1)];
 
 for idx=1:10
     logp_xn_given_zn = Gmm_logp_xn_given_zn(Data{idx}, phi);
     [~,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);
     path = LogViterbiDecode(logp_xn_given_zn, p_start, A);
     if idx <=5
         if length(find(path==control)) > length(path)/2
             predict_labels(idx) = control;
         else
             predict_labels(idx) = hunt;
         end
     else
        if length(find(path==hunt)) > length(path)/2
             predict_labels(idx) = hunt;
         else
             predict_labels(idx) = control;
         end
     end
 end
 
 TP=0;
 FP=0;
 TN=0;
 FN=0;
 for idx=1:10
 if idx <=5
         if predict_labels(idx) == labels(idx)
             TP = TP+1;
         else
             FP = FP +1;
         end
     else
         if predict_labels(idx) == labels(idx)
             TN = TN+1;
         else
             FN = FN +1;
         end
     end
 end
 
% Calculate accuracy
accuracy = (TP + TN) /10;
 
% Calculate precision
precision = TP / (TP + FP);
 
% Calculate recall (sensitivity)
recall = TP / (TP + FN);
 
%Calculate F1-score
 f1_score = 2 * (precision * recall) / (precision + recall);

Display the results
 fprintf('Accuracy: %.2f%%\n', accuracy * 100);
 fprintf('Precision: %.2f%%\n', precision * 100);
 fprintf('Recall: %.2f%%\n', recall * 100);
 fprintf('F1-score: %.2f%%\n', f1_score * 100);

% use the mse
path_control = 0;
for idx=1:5
    logp_xn_given_zn = Gmm_logp_xn_given_zn(Data{idx}, phi);
    [~,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);
    path = LogViterbiDecode(logp_xn_given_zn, p_start, A);
    path_control = path_control + path;
end
path_control = path_control / 5;

path_hunt = 0;
for idx=6:10
    logp_xn_given_zn = Gmm_logp_xn_given_zn(Data{idx}, phi);
    [~,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);
    path = LogViterbiDecode(logp_xn_given_zn, p_start, A);
    path_hunt = path_hunt + path;
end
path_hunt = path_hunt / 5;

labels = [zeros(5, 1); ones(5, 1)];
predict_labels = [zeros(10, 1)];
predict_final_logits = [zeros(10, 1)];
predict_logits = [zeros(10, 2)];

for idx=1:10
    logp_xn_given_zn = Gmm_logp_xn_given_zn(Data{idx}, phi);
    [~,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);
    path = LogViterbiDecode(logp_xn_given_zn, p_start, A);
    
    if mse(path-path_control) <= mse(path-path_hunt)
        predict_labels(idx) = 0;
    else
        predict_labels(idx) = 1;
    end
    
    predict_logits(idx, 1) = mse(path-path_control);
    predict_logits(idx, 2) = mse(path-path_hunt);
end
 
exp_sum  = sum(exp(predict_logits), 2);
predict_logits(:, 1) = exp(predict_logits(:, 1))./exp_sum;
predict_logits(:, 2) = exp(predict_logits(:, 2))./exp_sum;

for idx=1:10
    if predict_labels(idx)==0
        predict_final_logits(idx) = predict_logits(idx, 1);
    else
        predict_final_logits(idx) = predict_logits(idx, 2);
    end
end

ground_truth = labels;
edge_img = predict_final_logits;
thresh = 0:0.01:1;
FPR = zeros(size(thresh, 2));
TPR = zeros(size(thresh, 2));
for i = 1:length(thresh)
    pred = edge_img > thresh(i);
    TP = sum(ground_truth(:) & pred(:)); % true positive
    TN = sum(~ground_truth(:) & ~pred(:)); % true negative
    FP = sum(~ground_truth(:) & pred(:)); % false positive
    FN = sum(ground_truth(:) & ~pred(:)); % false negative
    FPR(i) = FP / (FP + TN);
    TPR(i) = TP / (TP + FN);
end
figure;
plot(FPR, TPR); xlabel('False Positive Rate'); ylabel('True Positive Rate'); title('ROC Curve');

TP=0;
FP=0;
TN=0;
FN=0;
for idx=1:10
    if idx <=5
        if predict_labels(idx) == labels(idx)
            TP = TP+1;
        else
            FP = FP +1;
        end
    else
        if predict_labels(idx) == labels(idx)
            TN = TN+1;
        else
            FN = FN +1;
        end
    end
end

% Calculate accuracy
accuracy = (TP + TN) /10;

% Calculate precision
precision = TP / (TP + FP);

% Calculate recall (sensitivity)
recall = TP / (TP + FN);

% Calculate F1-score
f1_score = 2 * (precision * recall) / (precision + recall);

% Display the results
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('Recall: %.2f%%\n', recall * 100);
fprintf('F1-score: %.2f%%\n', f1_score * 100);
