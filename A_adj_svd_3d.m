function Atb = A_adj_svd_3d(b,alpha,H_conj)
% Atb = A_adj_svd_3d(b,alpha,H_conj)
% Computes A' for space-varying low rank approximation to 3D deconvolution
% Atb = sum_r (alpha .* H_conj ** b)

Atb = sum(alpha .* real(ifft2(H_conj.*fft2(b))),4);