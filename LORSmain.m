function [B, L, mu] = LORSmain(Y,SNP)

[n, q] = size(Y);
p = size(SNP,2);


Omega0 = logical(1-isnan(Y));
Y(isnan(Y))=0; % put 0 for the missing values


%% first use soft-impute to select a reasonable lambda;
mask = (rand(n,q)>0.5);
Omega1 = Omega0 & mask;
Omega2 = Omega0 & (~mask);

maxRank = min(n,q)/2;
[Z, Err, rank_alpha, Znorm, Alpha] = softImpute(Y,[],Omega0, Omega1, Omega2, [],maxRank);
bestind_lam = min(Err)==Err;

lambda = Alpha(bestind_lam);

%% Get a good initialization of L
L = SVT(Y,lambda);

%% Set a sequence of rho
nrho = 20;
MaxRho = GetMaxRho(SNP,Y, L, Omega0); % use Omega0 to generate the Rho sequence
rhoseq = logspace(log10(MaxRho),log10(MaxRho*.05),nrho);
%rhoseq = linspace(MaxRho*0.5,MaxRho*.05,nrho);

tol = 1e-4;
rhoErr = zeros(nrho,1);
B = zeros(p,q);
for irho = 1:nrho    
    % here we use warm start: previous B to initilize the problem
[B, mu, L, Err] = LORS2(Y,SNP,L,Omega1,Omega2,B, rhoseq(irho),lambda,tol);
rhoErr(irho) = Err;
end
%% plot result
% figure
% plot(rhoErr,'.-');

%% use the best rho solve the optimization problem
bestind_rho = rhoErr==min(rhoErr);

if sum(sum((Omega0))) < n*q
    [B,L,mu] = LORS1(Y,SNP,[], Omega0, zeros(p,q), rhoseq(bestind_rho), lambda,tol);
else
    [B,L,mu] = LORS0(Y,SNP,rhoseq(bestind_rho),lambda,tol); % no missing
end
