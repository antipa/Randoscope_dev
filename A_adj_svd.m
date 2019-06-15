function x = A_adj_svd(H_conj, weights, y, pad)
% Ah_i = diag(W_i)ifft2(H_conj_i .* fft2(pad(y))


x = sum(weights .* real(ifft2(H_conj .* fft2(pad(y)))),3);