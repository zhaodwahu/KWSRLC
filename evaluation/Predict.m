function predict_target = Predict(Outputs,tau)
    predict_target = zeros(size(Outputs));
    num_class = size(Outputs,1);
    for l = 1:num_class
        predict_target(l,:) = sign(Outputs(l,:) - tau(1,l));
    end
end