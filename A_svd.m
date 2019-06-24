function y = A_svd(H, weights, x,crop)
Y = zeros(size(x));
for r = 1:size(H,3)
    X = fft2(weights(:,:,r).*x);
    Y = Y + (X.*H(:,:,r));
end
y = crop(ifft2(Y));

%y = real(crop(ifft2(sum(fft2(weights.*x).*H,3))));

