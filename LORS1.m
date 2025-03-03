function [B,mu,L] = LORS1(Y,X,L,Omega0,B,rho,lambda,tol)
% Y = mu + XB + L + e;
% min_{B,L} 0.5*||P_Omega(Y-XB-mu-L)||^2_F + rho ||B||_1 + lambada ||L||_*
% input:
% Y, X, L(for warm start), Omega0 
% Omega0 must be a logical variable
% Omega0
% 
% Output
% B, mu, L


% We use alternating strategy to solve this problem

[n,p] = size(X);
q = size(Y,2);

%B = zeros(p,q);
mu = zeros(1,q);

fval_old = 0;

maxIter = 100;
maxInnerIts = 50;

%initialization
if isempty(L)
    L = Y;
end
energy = inf;

for iter = 1:maxIter
    
    % softimpute algorithm for solving the low-rank matrix L
    for innerIts = 1:maxInnerIts
        % (a)i
        Z = Y-X*B-ones(n,1)*mu;
        C = Z.*Omega0 + L.*(1-Omega0);
        [U,D,V] = svd(C,'econ');
        VT = V';
        % soft impute
        d = diag(D);
        idx = find(d > lambda);
        W = diag( d(idx) - lambda );
        L = U(:,idx) * W * VT(idx,:);
        %Z = U(:,idx) * diag( d(idx) ) * VT(idx,:);
        % (a)ii
        Lnorm = sum(d(idx)-lambda);
        energy_old = energy;
        energy = lambda*Lnorm + norm(L(Omega0(:))-Z(Omega0(:)),'fro')/2;
        if abs(energy - energy_old) / energy_old < tol
            break;
        end
    end
    
   
    
    % compute B %call glmnet
    for j = 1:q
        %B(:,j) = LeastR(X(Omega(:,j),:),Y(Omega(:,j),j)-L(Omega(:,j),j),rho);
        opts.lambda = rho/sum(Omega0(:,j));
        opts.standardize = false;
        options = glmnetSet(opts); 
        fit = glmnet(X(Omega0(:,j),:),Y(Omega0(:,j),j)-L(Omega0(:,j),j),'gaussian',options);
        B(:,j) = fit.beta;
        mu(:,j) = fit.a0;
    end
    
    % convergence
    residual = Y - X*B - ones(n,1)*mu - L;
    dum = residual.*Omega0; dum = dum(:);
    fval = 0.5*(dum')*dum + rho*sum(abs(B(:))) + lambda*sum(abs(diag(W)));
    res = abs(fval-fval_old)/abs(fval_old+eps);
    
    fprintf('Iter %d, %f \n', iter, fval);
    if res < tol
        break
    else
        fval_old = fval;
    end
    
    
end




