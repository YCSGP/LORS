clear;
clc;

n = 100; % number of samples
q = 100; % number of genes
p = 50; % number of SNPs
k = 10;
snr = 1;
snrL = 3;

maf = .25;

tpr = [];
fpr = [];
auc = [];




SNP = binornd(1,maf,n,p);

V = randn(p, q).*(rand(p, q)>.99);


G = SNP*V;


hf = randn(n,k);
SIGMA = hf*hf';

MU = zeros(1,n);
L0 = mvnrnd(MU,SIGMA,q);
L0=L0';


L0 = L0*std(G(:))/std(L0(:))*snrL;
e = randn(size(L0))*std(G(:))/snr;
Y = G + L0 + e;

VarG = var(G(:));
VarL = var(L0(:));
Vare = var(e(:));

S0 = abs(V)>0;
% clear;
% load('I:\Code\SLR\SNP.mat');
% load('I:\Code\SLR\Y.mat');
% load('I:\Code\SLR\S0.mat');

mask = rand(p, q)>.9;
Y(mask) = NaN;



tic
[B, L, mu] = LORSmain(Y,SNP);
toc

[TPR, FPR, AUC] = roc_curve(B, S0);
 
tpr = [tpr,TPR];
fpr = [fpr,FPR];
auc = [auc,AUC];





