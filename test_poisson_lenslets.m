% Generate lenslet surface

make_lenslets_from_file;

fx = linspace(-1/2/px,1/2/px,Nx);
fy = linspace(-1/2/px, 1/2/px, Ny);


%%
[Fx, Fy] = meshgrid(fx,fy);
ior = 1.56;
[X, Y] = meshgrid(subx, suby);
R = 100;
sphase = exp(1i * 2*pi/lambda * R*sqrt(1-(X./R).^2 - (Y./R).^2));
CA = .9;
aper = sqrt(X.^2 + Y.^2)<=CA;
Ui = aper.*sphase.*exp(1i * 2*pi*(ior-1)/lambda * lsurf);
z = 8.9;
U = propagate2(Ui, lambda, z, Fx, Fy);
psf_poi = abs(U).^2;
imagesc(abs(U).^2)
axis image

h = imresize(psf_poi,1/4,'box');
h = hk;
[Ny, Nx] = size(h);
%h = double(data_in.psf) - .005;
%h = h/norm(h,'fro');
lf_strength = .0;
%lf_error_psf = imresize(real(ifft2(ifftshift(randn(size(U)).*(sqrt(Fx.^2 + Fy.^2)<.5)))),[Ny, Nx]);
lf_error_psf = imresize(double(imread(['/Users/nick.antipa/Documents/Diffusers/Miniscope/RandoscopeNanoscribe/RandoscopeNanoscribe/4_8_bck_1/','bck_4_8_ (172).tif'])),.25,'box');
lf_error_psf = lf_error_psf - min(lf_error_psf(:));
lf_error_psf = lf_error_psf / max(lf_error_psf(:));


h = h/max(h(:));%+ lf_strength * lf_error_psf/10;

pad = @(x)padarray(x,[floor(Ny/2), floor(Nx/2)],'both');
crop = @(x)x(floor(Ny/2)+1:end-floor(Ny/2),floor(Nx/2)+1:end-floor(Nx/2));

H = fft2(ifftshift(pad(h)));
H_conj = conj(H);

CtC = pad(ones(Ny, Nx,'like',h));

grad_x = @(x)real(ifft2(H_conj .* fft2( pad( x - y))));
A = @(x)crop(real(ifft2( H .* fft2(x))));
%%
%im_in = imread('cameraman.tif');
im_in = 255 - double(imread('/Users/nick.antipa/Downloads/969px-USAF-1951.svg.png'));
im_in = im_in(:,:,1);
im_gt = imresize(im_in,size(H));
meas = A(im_gt);
pcount = 1000;
lf_error_meas = imresize(double(imread('/Users/nick.antipa/Documents/Diffusers/Miniscope/RandoscopeNanoscribe/res_target_greentape_bck (180).tif')),.25,'box');
lf_error_meas = lf_error_meas - mean2(lf_error_meas);
lf_error_meas = lf_error_meas / max(lf_error_meas(:));
lf_meas_strength = .0;
meas = meas/max(meas(:)) + lf_meas_strength * lf_error_meas;
meas = imnoise(meas/max(meas(:))*pcount*1e-12,'poisson')*1e12;

n_var = 5;
read_noise = randn([Ny, Nx])*n_var;
meas = meas+read_noise;
meas = meas/max(meas(:));
imagesc(meas)
sett_file = '/Users/nick.antipa/Documents/developer/lensless/admm2d_settings.m';
%%


lf_strength = .005;
bg_const = .000;

psf_noise = h+lf_error_psf*max(h(:))*lf_strength + bg_const;
psf_nvar = 4;
pcount_psf = 1000;

psf_noise =psf_nvar * randn([Ny, Nx]) + imnoise(psf_noise/max(psf_noise(:))*pcount_psf*1e-12,'poisson')*1e12;
psf_noise = psf_noise/pcount_psf;
imagesc(psf_noise)
caxis([0 .1])
%plot(psf_noise)
roi = imresize(sqrt(X.^2 + Y.^2)>=400*px,.25,'box');

[bg_rm_opt, alpha_min, beta_min] = Miniscope3D_fit_bg(psf_noise, lf_error_psf, linspace(.001,.05,100), linspace(0,.01,50),roi);
alpha_min
beta_min
bg_dct = dct2(psf_noise);
dct_vec = bg_dct(:);
[~, i] = sort(abs(dct_vec),'descend');
dct_sort = dct_vec(i);
dct_sort(1:3) = 0;
dct_vec_thresh = zeros(size(dct_sort));
dct_vec_thresh(i) = dct_sort;
dct_zeros = reshape(dct_vec_thresh,size(bg_dct));

bg_dct(1:1,1:1) = 0;
psf_nodct = idct2(bg_dct);
imagesc(psf_nodct)

xhat(:,:,m) = admm2d_solver(meas,psf_noise , sett_file,.005);





