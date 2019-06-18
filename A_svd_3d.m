function b = A_svd_3d(v,alpha,H)
% b = A_svd_3d(v,alpha,H) 
% Computes the low rank approximation to space-varying 3D convolution with
% spatial weighting alpha and components with fourier transform H.
% i.e. b = Sum_r ( sum_z (h(x,y,z,r) ** alpha(x,y,z,r).*v(x,y,z)))
%
% alpha must be Ny-by-Nx-by-Nz-by-Nr
% v must be Ny-by-Nx-by-Nz
% H must be Ny-by-Nx-by-Nz-by-Nr

b = real(ifft2(sum(sum(H.*fft2(v.*alpha),3),4)));

