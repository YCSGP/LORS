function [B,L,mu] = LORS0(Y,X,rho,lambda,tol)
% Y = XB + L + e;
% min_{B,L} 0.5*||Y-XB-L||^2_F + rho ||B||_1 + lambada ||L||_*
% alternating strategy

[n,p] = size(X);
q = size(Y,2);

B = zeros(p,q);
mu = zeros(1,q);

maxIter = 100;

fval_old = 0;
for iter = 1:maxIter
    
    % compute L
    [U,W,V] = svd(Y-X*B-ones(n,1)*mu,'econ');
    VT = V';
    w = diag(W);
    ind = find(w>lambda);
    W = diag(w(ind)-lambda);
    L = U(:,ind)*W*VT(ind,:);
    
    % compute B
    for j = 1:q
        %B(:,j) = LeastR(X,Y(:,j)-L(:,j),rho);
        opts.lambda = rho/n;
        opts.standardize = false;
        options = glmnetSet(opts); 
        fit = glmnet(X,Y(:,j)-L(:,j),'gaussian',options);
        B(:,j) = fit.beta;
        mu(:,j) = fit.a0;
    end
    
    % convergence
    dum = Y-X*B-ones(n,1)*mu-L; dum = dum(:);
    fval = 0.5*(dum')*dum + rho*sum(abs(B(:))) + lambda*sum(abs(diag(W)));
    res = abs(fval-fval_old)/abs(fval_old+eps);
    
    fprintf('Iter %d, %f \n', iter, fval);
    if res < tol
        break
    else
        fval_old = fval;
    end
    
    
end




