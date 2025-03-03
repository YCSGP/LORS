function MaxRho = GetMaxRho(X, Y, L, Omega0)

q = size(Y,2);    
maxrho = zeros(q,1);
for i = 1:q
    maxrho(i) = max(abs(X'*((Y(:,i) - L(:,i)).*Omega0(:,i))));
    %maxrho(i) = max(abs(X'*(Y(:,i))));
end

MaxRho = max(maxrho);