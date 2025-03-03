# LORS
A Low-Rank representation and Sparse regression for eQTL mapping. This algorithm accounts for confounding factors such as unobserved covariates, experimental artifacts, and unknown environmental perturbations.

 doi: 10.1093/bioinformatics/btt075

please use demo to get start.

Suppose you have expression data Y and SNP data X
You can call function LORSmain as

[B, L, mu] = LORSmain(Y,SNP)

It will return you the estimated coefficient matrix B, the low-rank matrix L and intercept mu.
