function [optmParameter, modelparameter] =  initialization
    %% Optimization Parameters
    optmParameter.lambda1  = 10^-2;
    optmParameter.lambda2  = 2^0;  
    optmParameter.lambda3  = 2^-1; 
    optmParameter.lambda4  = 10^-4; 
    optmParameter.kernel_para      = 1; 
    
    optmParameter.rho      = 1.0; 
    optmParameter.maxIter  = 60;

      
    %% Model Parameters
    modelparameter.misRate = 0.8; % missing rate of positive class labels
    modelparameter.cv_num             = 5;
end



