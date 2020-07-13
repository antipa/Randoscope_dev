%Based on SNR analysis in Cossairt's paper
% Given some system y = Ax + n, with A is NxN in size.
% MSE of LLSE is inv(A'*A)*A'y for system modeled as
% sigma^2/N * trace(inv(A)'*inv(A)) 
% = sigma^2/N * trace(inv(A)*inv(A)') 
% Use whichever one doesn't blow up!
%
% The idea is to analyze the overdetermined linear system assuming a known
% support in sample space, and some total photon flux. In that framework,
% the measurement matrix will be the submatrix, As. In this case, we'll
% look at MSE of solving the normal equations. This requires a slight
% modification, but shouldn't be a big deal. We can analyze system:
%
% As*xs + n = ys
%
% = As'*As*sx + As'*n = As'*y
%
% This gives us the square, positive definite matrix As'*As to analyze,
% with noise defined by As'*n