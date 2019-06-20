% Read in PSF
% This is designed to work with measurements in a folder, then in a
% parallel folder, save the recons (i.e. measurements../recons/)
data_path = 'D:\Kyrollos\RandoscopeNanoscribe\measurements';   %<--folder where the measurements are

meas_name = 'res_target_1_MMStack_Pos0.ome.tif';    %<--- name of measurement
bg_name = 'bck_res_target_1_MMStack_Pos0.ome.tif';

meas_path = [data_path,'/',meas_name];
bg_path = [data_path,'/',bg_name];

psf_path = 'D:\Kyrollos\RandoscopeNanoscribe\RandoscopeNanoscribe\Miniscope3D\psf_svd_12comps_23z_240xy_20190619';
comps_path = [psf_path,'/comps.mat'];
weights_path = [psf_path,'/weights.mat'];

%%
h_in = load(comps_path);
h = h_in.comps_path;
clear h_in

weights_in = load(weights_path);
weights = weights_in.stack_weights_interp;
clear weights_in

%%

params.meas_depth = 41;    %If using 3D tiff, which slice was processed?
params.ds_z = 1;   %Temporal downsampling. Use 1 or 2. ds_t = 2 will reduce time points by 2x
params.meas_bias = 0;
init_style = 'zeros';   %Use 'loaded' to load initialization, 'zeros' to start from scratch. Admm will run 2D deconv, then replicate result to all time points
params.ds = 1;   %Lateral downsampling. Use 4 or 8. 
useGpu = 1;
params.psf_norm = 'none';   %Use max, slice, fro, or none

switch lower(params.psf_norm)
    case('max')
        h = h/max(sum(sum(h)));
    case('none')
    case('fro')
        h = h/norm(vec(h));
end




% Read in data
data_in = imresize(double(imread([data_path,'/', meas_name])) - params.meas_bias, 1/params.ds, 'box');
b = data_in/max(data_in(:));
% data_r = data_in(:,:,1);
% data_g = data_in(:,:,2);
% data_b = data_in(:,:,3);


Nx = 2*size(h,2);
Ny = 2*size(h,1);

%define crop and pad operators to handle 2D fft convolution
pad2d = @(x)padarray(x,[size(h,1)/2,size(h,2)/2],0,'both');
cc = gpuArray(size(h,2)/2+1):(3*size(h,2)/2);
rc = gpuArray(size(h,1)/2+1):(3*size(h,1)/2);
crop2d = @(x)x(rc,cc);


shutter_indicator = gpuArray(logical(hsvid_crop([Ny/2, Nx/2],params.nlines, pad2d)));   % Creates the rolling shutter indicator function in the padded 3d space

Nt = size(shutter_indicator,3);

if params.ds_t == 2
    if floor(Nt/2)*2 ~= Nt
        shutter_end = shutter_indicator(:,:,end);
        shutter_indicator = shutter_indicator(:,:,1:end-1);
    end
    shutter_indicator = gpuArray(single(.5*shutter_indicator(:,:,1:2:floor(Nt/2)*2) + .5*shutter_indicator(:,:,2:2:floor(Nt/2)*2)));
    if mod(size(shutter_indicator,3)/2,1)~=0
        %shutter_indicator = cat(3,shutter_indicator,shutter_end);
        shutter_indicator = shutter_indicator(:,:,1:end-1);   %Make it even
    end
end
%%


Nt = size(shutter_indicator,3);



if strcmpi(init_style, 'zeros')
    xinit_rgb = zeros(Ny, Nx, Nt,3);
elseif strcmpi(init_style,'loaded')
    xinit = imnormalized(:,:,:,cindex);
elseif strcmpi(init_style,'admm')
    xinit_2d = gpuArray(single(zeros(Ny, Nx, 3))); 
    xinit_rgb = zeros(Ny, Nx, Nt,3);
    for n = 1:3
        xinit_2d(:,:,n) = admm2d_solver(gpuArray(single(b(:,:,n))), gpuArray(single(h(:,:,n))),[],.001); 
        xinit_rgb(:,:,:,n) = params.nlines*repmat(gather(xinit_2d(:,:,n)),[1,1,Nt]);
        imagesc(2*xinit_2d/max(xinit_2d(:)))
    end
end


axis image
clear xinit_2d
%%

options.color_map = 'parula';
if params.ds == 8
    if params.nlines == 22
        options.stepsize = .3;
    elseif params.nlines == 6
        %options.stepsize = 4;
        options.stepsize = 4;

    elseif params.nlines == 3
        %options.stepsize = 8;
        options.stepsize = 24;
    elseif params.nlines == 1
        options.stepsize = 100;
    elseif params.nlines == 11
        options.stepsize = 1.5;
        

    else
        options.stepsize = 1.5;
    end
elseif params.ds == 12
    options.stepsize = 1;  %1 works for hande
elseif params.ds == 4
    if params.nlines == 1
        options.stepsize = 12;
    elseif params.nlines == 11
        options.stepsize = 3.5;
    elseif params.nlines == 22;
        options.stepsize = 7;
    end
end

options.convTol = 15e-10;

%options.xsize = [256,256];
options.maxIter = 2000;
options.residTol = 5e-5;
options.momentum = 'nesterov';
options.disp_figs = 1;
options.disp_fig_interval = 20;   %display image this often
options.xsize = size(h);
options.print_interval = 5;


h1 = figure(1);
clf
options.fighandle = h1;
nocrop = @(x)x;
options.known_input = 0;
xhat_rgb = zeros(Ny, Nx, Nt, 3);
%%

for cindex = params.colors(end)
    H = gpuArray(single(fft2(ifftshift(pad2d(h(:,:,cindex))))));
    Hconj = conj(H);


    % Setup forward op
    A = @(x)hsvidA(x, H, shutter_indicator, crop2d);
    % Adjoint
    Aadj = @(y)hsvidAadj(y, Hconj, shutter_indicator, pad2d);
    % Gradient

    if useGpu
        grad_handle = @(x)linear_gradient_b(x, A, Aadj, gpuArray(single(b(:,:,cindex))));
    else
        grad_handle = @(x)linear_gradient_b(x, A, Aadj, b(:,:,cindex));
    end
    
    %Prox
   % prox_handle = @(x)deal(x.*(x>=0), abs(sum(sum(sum(x(x<0))))));
    tau1 = gpuArray(.000006);   %.000005 works pretty well for v1 camera, .0002 for v2
    tau_iso = gpuArray(.25e-4);
    tau2 = .1;
    %prox_handle = @(x)deal(1/3*(x.*(x>=0) + soft(x, tau2) + tv3dApproxHaar(x, tau1)), TVnorm3d(x));
    prox_handle = @(x)deal(1/2*(max(x,0) + tv3dApproxHaar(x, tau1, 30)), tau1*hsvid_TVnorm3d(x));
    TVpars.epsilon = gpuArray(1e-7);
    TVpars.MAXITER = gpuArray(100);
    TVpars.alpha = gpuArray(.3);
    %prox_handle = @(x)deal(hsvid_TV3DFista(x, tau_iso, 0, 10, TVpars) , tau_iso*hsvid_TVnorm3d(x));

    xinit = gpuArray(squeeze(xinit_rgb(:,:,:,cindex))/30);
    if strcmpi(init_style, 'zeros')
        xinit = gpuArray(zeros(Ny, Nx, Nt));
    end
    if useGpu
        xinit = gpuArray(single(xinit));
        [xhat, f2] = proxMin(grad_handle,prox_handle,xinit,gpuArray(single(b(:,:,cindex))),options);
    else 
        [xhat, f2] = proxMin(grad_handle,prox_handle,xinit,b(:,:,cindex),options);
    end
    xhat_rgb(:,:,:,cindex) = gather(xhat);
    clear xhat
end

%%


datestamp = datetime;
date_string = datestr(datestamp,'yyyy-mmm-dd_HHMMSS');
save_str = ['../recons/',date_string,'_',meas_name(1:end-4)];
full_path = fullfile(data_path,save_str);
mkdir(full_path);


%%
imout = xhat_rgb/prctile(xhat_rgb(:),99.99);
imbase = meas_name(1:end-4);
mkdir([full_path, '/png/']);
filebase = [full_path, '/png/', imbase];
out_names = {};
for n= 1:size(imout,3)
    out_names{n} = [filebase,'_',sprintf('%.3i',n),'.png'];
    imwrite(squeeze(imout(:,:,n,:)),out_names{n});
    fprintf('writing image %i of %i\n',n,size(xhat_rgb,3))
end

fprintf('zipping...\n')
zip([full_path, '/png/', imbase],out_names)
fprintf('done zipping\n')
%%
fprintf('writing .mat\n')
save([full_path,'/',meas_name(1:end-4),'_',date_string,'.mat'], 'tau_iso','TVpars','xhat_rgb', 'options', 'h', 'b','params','options','-v7.3')
fprintf('done writing .mat\n')