function [model] = KWSRLC(X,Y,optmParameter)

lambda1=optmParameter.lambda1;
lambda2=optmParameter.lambda2;
lambda3=optmParameter.lambda3;
lambda4=optmParameter.lambda4;

rho=optmParameter.rho;
kernel_para=optmParameter.kernel_para;

max_iter=optmParameter.maxIter;

%% initializtion
[n,~]=size(X);
num_class=size(Y,2);

P=zeros(num_class,num_class);
E=zeros(n,num_class);
kernel_type='rbf';
Kx = kernelmatrix(kernel_type,X',X',kernel_para); 

A=(Kx+rho*speye(n))\(Y);
options = [];
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 10;  % nearest neighbor
options.WeightMode = 'HeatKernel';
options.t = 1;
S = constructW(X,options);
format long

D = full(sparse(1:n, 1:n, sum(S))); 
L = eye(n)-(D^(-1/2) * S * D^(-1/2));

M = ones(n,n)*(1/n)*(-1) + speye(n);
MLM=M*L*M;
IsConverge = 0; iter = 1; 

    while (IsConverge == 0&&iter<max_iter+1)
        
        %update H
        H=(Kx*A*P' + lambda2*(Y-E)*P')*pinv(P*P' + lambda2*P*P'+eye(num_class)*1e-6);

        %update A
        A=(Kx+lambda1*speye(n)-lambda4* MLM*Kx)\(H*P);

        % update P
        P=updateP(H,Kx,A,Y,E,lambda2);
        % update E
        G = Y-H*P;
        [E] = solve_l1l2(G,lambda3/(lambda2));
        % check convergence
       thrsh = 1e-5;
        if(norm(Y-H*P-E,inf)<thrsh && norm(Kx*A-H*P,inf)<thrsh)
            IsConverge = 1;
        end
        iter = iter + 1;
    end
        model.A=A;
        model.Kx=Kx;
end
function P=updateP(H,Kx,A,Y,E,lambda1)

        F=H'*Kx*A+lambda1*H'*(Y-E);
        [U,~,V] = svd (F,'econ'); 
        P=U*V';
end


