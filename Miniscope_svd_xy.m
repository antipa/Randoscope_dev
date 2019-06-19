function [comps, weights_interp, weights,shifts,yi_reg_out] = Miniscope_svd_xy(stack,icenter,rnk,varargin)
% [comps, weights,shifts] = Miniscope_svd_xy(stack,center_id,rnk)
% Takes in a background subtracted stack. Stack does not need to be on a
% grid and will be blindly registerd with cross correlation. Registration
% is performed relative to center_id location. 
% SVD up to rank rnk is done after registration.
% 'boundary_condition' : 'circular' (defualt) or 'zero'. 'circular' uses no
% padding/cropping, 'zero' pads and crops to avoid wrapping (good if
% cropping is happening'). Default is 'circular'
% Returns 3d arrays comps and weights. comps
% contains the psf components, and weights contains the spatially-varying
% weights_iterp, upsampled to the same grid size as the inpus stack.
% Returns weights, the SVD weighting for each measurement (not on a
% grid--this is index-aligned with shifts)
% Also returns cell array of shifts found in registration
% Returns yi_reg_out, an 8-bit image of each psf after registration. This
% is just for visual inspection!

p = inputParser;
addParameter(p,'boundary_condition','circular')
parse(p,varargin{:})
params = p.Results;

[Ny, Nx] = size(stack(:,:,1));
pad2d = @(x)padarray(x,[Ny/2,Nx/2],'both');
fftcorr = @(x,y)(ifft2(fft2(pad2d(x)).*conj(fft2(ifftshift(pad2d(y))))));

findpeak_id = @(x)find(x == max(x(:)));

M = size(stack,3);
Si = @(x,si)circshift(x,si);

pr = Ny + 1;
pc = Nx + 1; % Relative centers of all correlations


yi_reg = 0*stack;   %Registered stack
yi_reg_out = uint8(yi_reg);
switch lower(params.boundary_condition)
    case('circular')
        pad = @(x)x;
        crop = @(x)x;
    case('zero')
        pad = @(x)padarray(x,[Ny/2,Nx/2],'both');
        crop = @(x)x(Ny/2+1:3*Ny/2,Nx/2+1:3*Nx/2,:);
end

% Normalize the stack first
stack_norm = zeros(1,M);
for m = 1:M
    stack_norm(m) = norm(stack(:,:,m),'fro');
    stack(:,:,m) = stack(:,:,m)/stack_norm(m);
    
end
si = cell(1,M);
% Do fft registration
fprintf('Removing background and hot pixels...\n')

stack_dct = stack;

for n = 1:size(stack_dct,3)
    im = stack(:,:,n);
    bg_dct = dct2(remove_hot_pixels(im,3,.0001));
    bg_dct(1:20,1:20) = 0;

    stack_dct(:,:,n) = idct2(reshape(bg_dct,size(im)));

    
end

fprintf('done\n')

fprintf('registering...\n')

for m = 1:M
    
    %[r,c] = ind2sub(2*[Ny, Nx],findpeak_id(fftcorr(remove_hot_pixels(stack(:,:,m),3,.0001),stack(:,:,icenter))));
    [r,c] = ind2sub(2*[Ny, Nx],findpeak_id(fftcorr(stack_dct(:,:,m),stack_dct(:,:,icenter))));
    si{m} = [-(r-pr),-(c-pc)];
    yi_reg(:,:,m) = crop(Si(pad(stack(:,:,m)),si{m}));
    yi_reg_out(:,:,m) = uint8(yi_reg(:,:,m)/max(max(yi_reg(:,:,m)))*255);
    
end
fprintf('done registering\n')

fprintf('creating matrix\n')
ymat = zeros(Ny*Nx,M);
for m = 1:M
    ymat(:,m) = vec(yi_reg(:,:,m));
    
end
fprintf('done\n')

fprintf('starting svd...\n')
tic
[u,s,v] = svds(ymat,rnk);
t_svd = toc;
fprintf('svd took %.2f seconds \n',t_svd)

comps = reshape(u,[Ny, Nx,rnk]);
vt = v';

weights = zeros(M,rnk);
for m = 1:M
    for r = 1:rnk
        weights(m,r) = s(r,r)*vt(r,m);
    end
end
si_mat = reshape(cell2mat(si)',[2,M]);
xq = -Nx/2+1:Nx/2;
yq = -Ny/2+1:Ny/2;
[Xq, Yq] = meshgrid(xq,yq);

weights_interp = zeros(Ny, Nx,rnk);
fprintf('interpolating...\n')
for r = 1:rnk
    interpolant_r = scatteredInterpolant(si_mat(2,:)', si_mat(1,:)', weights(:,r),'natural','nearest');
    weights_interp(:,:,r) = rot90(interpolant_r(Xq,Yq),2);
end
fprintf('done')

shifts = si;


return





