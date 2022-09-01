function [ tau,  currentResult] = TuneThreshold( output, target)
% Tune the threshold for multi-label learning algorithms on the training
% data with one evaluation metric
    fprintf('- Tune threshold for multi-label classification\n');
    [num_class] = size(target,1);
    TotalNums = 50;
    
    min_score = min(min(output));
    max_score = max(max(output));
    step = (max_score - min_score)/TotalNums;
    tau_range = min_score:step:max_score;
    
    tau = zeros(1,num_class);
    currentResult = tau;
    for t = 1:length(tau_range)
        threshold = tau_range(t);

            thresholds = threshold*ones(size(output));
            predict_target = sign(output - thresholds);

            tempResult = evaluateOneMetric(target, predict_target);

            if tempResult > currentResult(1,1)
                currentResult(1,1) = tempResult;
                tau(1,1) = threshold;
            end      
    end
    tau = tau(1,1)*ones(1,num_class);
end


function  Result = evaluateOneMetric(target, predict_target)
% predict_target
% target
%   
    HammingScore = 1 - Hamming_loss(predict_target,target);
    Result = HammingScore;

end