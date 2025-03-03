function L = SVT(Y, lambda)
%min 0.5*||Y-L||^2_F + lambda*||L||_* 
% Closed-form solution by so-called "Soft-Thresholding"

    [U,W,V] = svd(Y,'econ');
    VT = V';
    w = diag(W);
    ind = find(w>lambda);
    W = diag(w(ind)-lambda);
    L = U(:,ind)*W*VT(ind,:);
    
    
  
