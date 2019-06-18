%Basic pipeline from calibration through to final image:

Ny = 480;
Nx = 640;
Nz = 30;
Nr = 10;


h = rand(Ny,Nx,Nz,Nr);
alpha = rand(Ny,Nx,Nz,Nr);

H = fft2(h);
H_conj = conj(H);

v = rand(Ny,Nx,Nz);

b = rand(Ny,Nx);
% profile off
% profile('-memory','on');
% 
% profile on;

tic
b_for = A_svd_3d(v,alpha,H);
toc

tic
v_for = A_adj_svd_3d(b,alpha,H_conj);
toc

% profile viewer

vec(b)'*vec(b_for) - vec(v)'*vec(v_for)

%%
ds = 4;
%ystack = read_im_stack('/Users/nick.antipa/Documents/Diffusers/Miniscope/RandoscopeNanoscribe/Shift_Varying/meas',ds);
%data_dir = '/Users/nick.antipa/Documents/Diffusers/Miniscope/RandoscopeNanoscribe/Shift_variance_june2019';
data_dir = '/Users/nick.antipa/Documents/Diffusers/Miniscope/RandoscopeNanoscribe/Shift_variance_june2019/Dense/'
xystackpath = [data_dir,'/shifted_psf_1_2x_3dtiff.tif'];
ystack = double(read_tiff_stack(xystackpath,ds/2));
%%
bgpath = [data_dir,'/shifted_psf_background.tif']
bg = imresize(double(imread(bgpath)),1/ds,'box');
%%
timing_ = tic;
[comps, weights,weights_interp,shifts,yi_reg_out] = Miniscope_svd_xy(ystack-bg,182,12);
toc(timing_)