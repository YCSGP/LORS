%% function for soft-impute
function [Z, Err, rank_alpha, znorm, Alpha] = softImpute(X,Z,Omega0,Omega1,Omega2,alpha0,maxRank)
%
% This program implements the soft-impute algorithm followed by
% postprocessing in the Matrix completion paper Mazumder'10 IJML
% min || X - Z ||_Omega + \alpha || Z ||_Nulear
% \alpha is decrease from alpha0 to the minima value that makes rank(Z) <= maxRank

% X is the incomplete matrix
% maxRank is the desired rank in the constraint
% Omega0 is the mask with value 1 for data and 0 for missing part
% Omega1 is the mask for training
% Omega2 is the mask for testing

X_0 = X.*Omega0;
if isempty(Z)
    Z = X_0;
end

if isempty(alpha0)
    [UU,D,VV] = svd(X_0,'econ'); 
    alpha0 = D(2,2);
end

if isempty(maxRank)
    maxRank = -1;
end
% parameters
eta = 0.9;
epsilon = 1e-4;
maxInnerIts = 50;
%% trivial
% no rank constraint
% if maxRank >= min(size(X_0))
%     Z = X_0;
%     [UU,D,VV] = svd(Z,'econ');
%     Znorm = sum(diag(D));
%     alpha = 0;
%     return;
% end
% % no observation
% if sum(Omega0(:)) == 0
%     % no data
%     Z = zeros(size(X_0));
%     Znorm = 0;
%     alpha = alpha0;
%     return;
% end
%% soft-impute
% 1. initialize
%outIts = 0;
alpha = alpha0;

%npoint = 50;
%Alpha = logspace(log10(alpha0),log10(D(rankD,rankD)),npoint);

%ii = 1;
%alpha = Alpha(ii);

Err = [];
rank_alpha = [];
%sigma =[];
znorm = [];
Alpha = [];

% 2. Do for alpha = alpha0 > alpha_1 > alpha_2 > ... > alpha_maxRank
disp('begin soft-impute iterations');
while 1
    %outIts = outIts + 1;
    energy = inf;
    for innerIts = 1:maxInnerIts
        % (a)i
        C = X.*Omega1 + Z.*(1-Omega1);
        [U,D,V] = svd(C,'econ');
        VT = V';
        % soft impute
        d = diag(D);
        idx = find(d > alpha);
        Z = U(:,idx) * diag( d(idx) - alpha ) * VT(idx,:);
        %Z = U(:,idx) * diag( d(idx) ) * VT(idx,:);
        % (a)ii
        Znorm = sum(d(idx)-alpha);
        energy_old = energy;
        energy = alpha*Znorm + norm(Z(Omega1(:))-X(Omega1(:)),'fro')/2;
        if abs(energy - energy_old) / energy_old < epsilon
            break
        end
    end
    % 
    e = X.*Omega2 - Z.*Omega2; % evaluate test error on the testing entries
    err2 = e(:)'*e(:);
    %sigma = [sigma, std(e(:))];
    Err = [Err, err2];
    znorm = [znorm, Znorm];
    % check termination condition of alpha
    k = length(idx); % rank of Z
    
    rank_alpha = [rank_alpha, k];
    Alpha = [Alpha, alpha];
    
    disp(['alpha = ' num2str(alpha) ';    rank = ' num2str(k) ';  number of iteration: ' num2str(innerIts)]);
    if k <= maxRank && alpha > 1e-3 
        alpha = alpha*eta;
        %ii=ii+1;
        %alpha = Alpha(ii);
    else
        break;      
    end    
end
