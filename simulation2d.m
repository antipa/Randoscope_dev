%load('D:\Kyrollos\PSFOptimization\fish_double.mat');        % load one in to see the size
%load('D:\Kyrollos\PSFOptimization\psf_abermulti.mat');   
%load('D:\Kyrollos\PSFOptimization\psf_montebest.mat');   
%load('D:\Kyrollos\PSFOptimization\psf_monteworst.mat');   
% load in image
%vol =fish;%flip(fish,3);
%vol=read_tiff_stack('C:\Users\kyrollos\Downloads\Miniscope3D-master\dataforrebuttal\fibers.tif',1);
%vol=vol(250:end,1:700,:);
%vol=imresize3(vol,[486,648,10]);
%noise=zeros(512,512,10);

% NAA: I ran for planes 5 and 36 -- is that right? 
psf_type = 'randoscope';
z=40;  %1 and 40
switch lower(psf_type)
    case('uni')

        in = load('D:\Randoscope\dataforrebuttal\newpsf\psf_uni_ds.mat');   
        psf_stack = permute(in.psf_noaber_uni_mid_ds, [2 3 1]);
        psf=psf_stack(:,:,z);%6use 20 and 60
    case('randoscope')
        in = load('D:\Randoscope\dataforrebuttal\newpsf\psf_multi_ds.mat');   
        psf_stack = permute(in.psf_noaber_multi_mid_ds, [2 3 1]);

        psf=psf_stack(:,:,z);% use 20 and 60, end-5 for noise sims
    case('regular')
        in = load('D:\Randoscope\dataforrebuttal\newpsf\psf_reg_ds.mat');
        psf_stack = permute(in.psf_noaber_reg_mid_ds, [2 3 1]);
        psf = psf_stack(:,:,z);  %use 72 and 32
    case('miniscope')
        generateminiscopepsf
        psf = miniscope_psf;
end


h=psf/sum(psf(:));
%px = 
%h=miniscope_psf;
res_target=imread('C:\Users\kyrollos\Downloads\Miniscope3D-master\dataforrebuttal\Asset 6.png');
res_target=rgb2gray(res_target);
res_target=double(res_target);
res_target=imresize(res_target,[512, 512]);
res_target=res_target/max(res_target(:));
res_target(res_target<0.2)=0;
figure(1)
imagesc(h)

axis image


% read noise
mu = 0.0*2;
a = 0.0063/2;
read_noise = a.*randn(512*512,1) + mu;
read_noise=reshape(read_noise,[512,512]);
%%
% ynoise=ynoise*255;
% y=uint8(y);
% vol=vol(200:end,1:800,:);
% vol=imresize3(vol,[486,648,20]);

data_path='C:\Users\kyrollos\Downloads\Miniscope3D-master\dataforrebuttal\';

psf_path = 'D:\Antipa\RandoscopeMono_PSFs_Nov\2_5umPSF';
%psf_path = 'T:\Antipa\Randoscopev2_PSFs\20190912_recalibration\SVD_2p5_um_PSF_5um_1_green_channel';
comps_path = [psf_path,'\2p5_4xds_rank12_zds4x_components.mat'];
weights_path = [psf_path,'\2p5_4xds_rank12_zds4x_weights.mat'];



%%
params.data_tiff_format = 'time';   %Use 'time' if tiff stacks are at the same location over time, use 'z' if they are z stacks'
params.tiff_color = 2;    %use 'rgb' or 'mono'. Use number (1,2,3) for r,g, or b only
params.meas_depth = 82;    %If using 3D tiff or list of files, which slice was processed?

params.ds_z = 1;   %z downsampling ratio
params.meas_bias = 0;

params.ds = 1;  % Global downsampling ratio (i.e.final image-to-sensor ratio)
params.ds_psf = 1;   %PSf downsample ratio (how much to further downsample -- if preprocessing included downsampling, use 1)
params.ds_meas = params.ds;   % How much to further downsample measurement?
params.z_range = 1:10; %Must be even number!! Range of z slices to be solved for. If this is a scalar, 2D. Use this for subsampling z also (e.g. 1:4:... to do every 4th image)
params.rank = 12;
useGpu = 1; %cannot fit ds=2 on gpu unless we limit z range!!!!
params.psf_norm = 'fro';   %Use max, slice, fro, or none

%meas_name = ['real_res_target_10um_1_MMStack_Img_',num2str(params.meas_depth),'_000_000.ome.tif'];    %<--- name of measurement
%bg_name = ['bck_real_res_target_10um_1_MMStack_Img_',num2str(params.meas_depth),'_000_000.ome.tif'];


% Normalize weights to have maximum sum through rank of 1




%define crop and pad operators to handle 2D fft convolution
pad2d = @(x)padarray(x,[size(h,1)/2,size(h,2)/2],0,'both');
ccL = size(h,2)/2+1;
ccU = 3*size(h,2)/2;
rcL = size(h,1)/2+1;
rcU = 3*size(h,1)/2;

%cc = gpuArray((size(h,2)/2+1):(3*size(h,2)/2));
%rc = gpuArray((size(h,1)/2+1):(3*size(h,1)/2));
crop2d = @(x)x(rcL:rcU,ccL:ccU);
crop3d=@(x)x(rcL:rcU,ccL:ccU,:);

H = fft2(ifftshift(ifftshift(pad2d(h),1),2));
Hconj = conj(H);
if useGpu
    h=gpuArray(h);
    H = gpuArray(H);
    Hconj = gpuArray(Hconj);
end







%%
% noise=zeros(486,648,20);
% b = 0.04;
% a = 0.0063;
% y = a.*randn(486*648*20,1) + b;
% y=reshape(y,[486,648,20]);

%%
%b = (A_svd_3d(single((vol)),(weights),H));

b_original = double(A_lensless_2d(double(h),double(res_target),pad2d,crop2d,1));
photon_count=100;
%b=b/max(b(:));
b_photoncount = b_original*photon_count;

b_noise = imnoise(b_photoncount*1e-12,'poisson')*1e12;
b_noise=read_noise*max(b_photoncount(:))+b_noise;
% b_noise=b_noise/max(b_noise(:))+read_noise;
% b_noise=b_noise/max(b_noise(:));
% b_noise(b_noise<0)=0;
b=b_noise;
psnr_raw = psnr(double(gather(b/photon_count)),double(gather(res_target)),1);
%b=b/max(b(:));

%%
% b_miniscope = double(A_lensless_2d(double(miniscope_psf),double(res_target),pad2d,crop2d,1));
% % b_miniscope(b_miniscope<0)=0;
% % b_miniscope=b_miniscope/max(b_miniscope(:));
% 
% b_photoncount_miniscope = b_miniscope*photon_count;
% b_noise_miniscope = imnoise(b_photoncount_miniscope*1e-12,'poisson')*1e12;
% b_noise_miniscope=read_noise*max(b_photoncount_miniscope(:))+b_noise_miniscope;
% % b_noise_miniscope=b_noise_miniscope/max(b_noise_miniscope(:))+read_noise;
% b_noise_miniscope=b_noise_miniscope/max(b_noise_miniscope(:));
% % b_noise_miniscope(b_noise_miniscope<0)=0;

%% reconstruct
tau_list = logspace(-2,1,30);
im_tag = '2DResTarget_miniscope3d_multifocal_photocount_100_outfocus';
psnr_best = 0;
psnr_list = []
for i=1:length(tau_list)
options.maxIter = 1000;

%params.tau1=0.12*1;%tau/10%tau(i);
params.tau1 = tau_list(i);
tau=params.tau1;
init_style = 'zeros';   %Use 'loaded' to load initialization, 'zeros' to start from scratch. Admm will run 2D deconv, then replicate result to all time points
params.data_format='tif';
if numel(size(h)) == 3
    [Ny, Nx, Nr] = size(h);
    Nz = 1;
else
    [Ny, Nx] = size(h);
    Nz=1;
end



if strcmpi(init_style, 'zeros')
    xinit = zeros(Ny, Nx, Nz);
elseif strcmpi(init_style,'loaded')
    if ~exist('xinit')
        xinit = zeros(Ny,Nx,Nz);
    else
        xinit = xhat_out(:,:,:);
    end
elseif strcmpi(init_style,'admm')
    xinit_2d = gpuArray(single(zeros(Ny, Nx, 3)));

    for n = 1:3
        xinit_2d(:,:,n) = admm2d_solver(gpuArray(single(b(:,:,n))), gpuArray(single(h(:,:,n))),[],.001);

        %imagesc(2*xinit_2d/max(xinit_2d(:)))
    end
end







options.color_map = 'parula';



options.convTol = 1e-9;

%options.xsize = [256,256];
options.residTol = 5e-5;
options.momentum = 'nesterov';
options.disp_figs = 1;
options.disp_fig_interval = 100;   %display image this often
if Nz == 1
    options.xsize = [Ny, Nx];
else
    options.xsize=[Ny, Nx, Nz];
end
options.print_interval = 100;

figure(2)
clf
imagesc(b)
axis image

h1 = figure(1);
clf
options.fighandle = h1;
nocrop = @(x)x;
options.known_input = 0;





large = 0;
if Nz > 1
    if large == 0
        A = @(x)crop2d(A_svd_3d(single(pad2d(x)),pad2d(weights),H));

        Aadj = @(y)crop3d(A_adj_svd_3d(pad2d(y), pad2d(weights), Hconj));
%         A = @(x)(A_svd_3d(single((x)),(weights),H));
% 
%         Aadj = @(y)(A_adj_svd_3d((y),(weights), Hconj));
    else
        weights=gpuArray(weights);
        H = gpuArray(H);
        Hconj = gpuArray(Hconj);
        b = gpuArray(single(b));
        A = @(x)A_svd_3d_large(x,weights,H); %
        Aadj = @(y)A_adj_svd_3d_large(y, weights, Hconj);
    end
elseif Nz == 1
    A = @(x)A_lensless_2d(h,single(x),pad2d,crop2d,1);
    Aadj = @(y)A_adj_lensless_2d(h,y,crop2d,pad2d,1);
end




       %options.stepsize = .1e-3; for ds=4
if params.ds == 1
    if strcmpi(params.psf_norm ,'fro')
        if Nz == 18
            options.stepsize = 3e-3;
        elseif Nz == 12
            options.stepsize = .4e-2;
            fprintf('foo\n')
        elseif Nz == 14
            options.stepsize = 4e-3;
        elseif Nz == 20
            if params.rank  == 12
                options.stepsize = 3e-3;
            elseif params.rank == 8
                options.stepsize = 1e-3;
            elseif params.rank == 18
                options.stepsize = 4e-3;
            else 
                options.stepsize = 3e-3;
              


            end
        elseif Nz>20
            options.stepsize = .014;  %015 is nice?
        else
            options.stepsize = 1;
            %options.stepsize=1.8/max(max(abs(fft2(h))));   %Nick added this -- it's good for 2D only! 
        end

    else
        options.stepsize = 3e-6;
    end

elseif params.ds == 2
    options.stepsize = 0.7e-3;
end
%.3e-4 is good for waterbears
%params.tau1 = options.stepsize*.1e-3; %was 0.5e-7   %.000005 works pretty well for v1 camera, .0002 for v2
params.tau_soft = options.stepsize * 3e-3;
tau_iso = (.25e-4);
params.z_tv_weight = 1;    %z weighting in anisotropic TV
tau2 = .001;  %Auxilliary
TVnorm3d = @(x)sum(sum(sum(abs(x))));


if useGpu

    grad_handle = @(x)linear_gradient_b(x, A, Aadj, gpuArray(single(b)));

    params.tau1 = gpuArray(params.tau1);
    params.tau_soft = gpuArray(params.tau_soft);
    tau_iso = gpuArray(tau_iso);
    params.z_tv_weight = gpuArray(params.z_tv_weight);
    options.stepsize = gpuArray(options.stepsize);

else
    if ~large
        grad_handle = @(x)linear_gradient_b(x, A, Aadj, single(b));
    else
        grad_handle = @(x)linear_gradient_large(x,A,Aadj,gpuArray(single(b)));
    end

end

%Prox
%prox_handle = @(x)deal(x.*(x>=0), abs(sum(sum(sum(x(x<0))))));



%prox_handle = @(x)deal(1/3*(x.*(x>=0) + soft(x, tau2) + tv3dApproxHaar(x, params.tau1)), TVnorm3d(x));




if ~strcmpi(params.data_format,'mat')
    if Nz>1
        prox_handle = @(x)deal(1/2*(max(x,0) + (tv3d_iso_Haar((x), params.tau1, params.z_tv_weight))), params.tau1*TVnorm3d(x));
    elseif Nz == 1
        prox_handle = @(x)deal(.5*tv2d_aniso_haar(x,params.tau1*options.stepsize) + ...
            .5*max(x,0), params.tau1*options.stepsize*TVnorm(x));
    end
else
%             prox_handle = @(x)deal(.5*(soft(x,params.tau_soft) +...
%                 tv3d_iso_Haar(x, params.tau1, params.z_tv_weight)), ...
%                 params.tau1*TVnorm3d(x));
    prox_handle = @(x)deal(soft(tv3d_iso_Haar(x, params.tau1, params.z_tv_weight),params.tau_soft), ...
        params.tau1*TVnorm3d(x));
    %prox_handle=@(x)deal(soft(x,params.tau_soft),params.tau_soft*sum(abs(vec(x))));
end
TVpars.epsilon = 1e-7;
TVpars.MAXITER = 100;
TVpars.alpha = .3;
%prox_handle = @(x)deal(hsvid_TV3DFista(x, tau_iso, 0, 10, TVpars) , hsvid_TVnorm3d(x));

if strcmpi(init_style, 'zeros')
    xinit = zeros(Ny, Nx, Nz);

end

if useGpu

    TVpars.epsilon = gpuArray(TVpars.epsilon);
    TVpars.MAXITER = gpuArray(TVpars.MAXITER);
    TVpars.alpha = gpuArray(TVpars.alpha);
    xinit = gpuArray(single(xinit));
    success = false;
    while success == false   %This shouldn't be necessary, but it deals with restarting when GPU runs OOM
        try
            [xhat, f2] = proxMin(grad_handle,prox_handle,xinit,gpuArray(single(b)),options);
            success = true;
        catch
            success = false;
        end
    end

else
    if large
        xinit = gpuArray(xinit);
    end
    [xhat, f2] = proxMin(grad_handle,prox_handle,xinit,b,options);
end

psnr_list(i) = psnr(double(gather(xhat/photon_count)),double(gather(res_target)),1);
if psnr_list(i) > psnr_best
    psnr_best = psnr_list(i);
    recon_best = gather(xhat);
    tau_best = params.tau1;
    stepsize_best = options.stepsize;
end

figure(3)
clf
subplot(1,2,1)
imagesc(recon_best)
title(sprintf('best, psnr: %.2f',psnr_best))
axis image
caxis([0 photon_count])

subplot(1,2,2)
imagesc(double(gather(xhat)))
axis image
title(sprintf('current, psnr: %.2f',psnr_list(i)))
caxis([0 photon_count])
drawnow





datestamp = datetime;
tiff_string = sprintf('%03d',num2str(i));
date_string = datestr(datestamp,'yyyy-mmm-dd_HHMMSS');
save_str = ['../recons/',date_string,'_','_',im_tag,'_tau_number',num2str(i)];
full_path = fullfile(data_path,save_str);
mkdir(full_path);



imout = gather(xhat/prctile(xhat(:),100*(numel(xhat)-10)/numel(xhat)));   %Saturate only 10 pixels
xhat_out = gather(xhat);
params.tau1 = gather(params.tau1);
params.tau_soft = gather(params.tau_soft);
imbase = 'zebra';
err = immse(xhat_out/max(xhat_out(:)),single(res_target));
mkdir([full_path, '/png/']);
filebase = [full_path, '/png/', imbase];
f_out = gather(f2);
out_names = {};
for n= 1:size(imout,3)
    out_names{n} = [filebase,'_',sprintf('Z_%.3i_T_',params.z_range(n)),...
        tiff_string,'_',im_tag,'.png'];
    imwrite(imout(:,:,n),out_names{n});
    fprintf('writing image %i of %i\n',n,size(xhat,3))
end

fprintf('zipping...\n')
zip([full_path, '/png/', imbase],out_names)
fprintf('done zipping\n')

fprintf('writing .mat\n')
options.fighandle = []

save([full_path,'/','recon_',date_string,'_',im_tag,'_',tiff_string,'tau_',num2str(params.tau1),'_mse_',num2str(err),'.mat'], 'tau_iso','TVpars','xhat_out', 'options', 'comps_path','weights_path', 'b','params')
fprintf('done writing .mat\n')
end

%% Save images
pth = 'C:\\Users\\kyrollos\\Downloads\\Miniscope3D-master\\dataforrebuttal\\';
imwrite(uint8(min(recon_best/photon_count,1)*(length(colormap(parula))-1)),colormap(parula),...
    [pth,psf_type,sprintf('_recon_Nicknewpsf1_outfocus_%iphotons_PSNR_%.4f_a_%.4f',...
    photon_count,psnr_best,a),'.png'])

imwrite(uint8(min(gather(b)/photon_count,1)*(length(colormap(parula))-1)),colormap(parula),...
    [pth,psf_type,sprintf('_raw_Nicknewpsf1_outfocus_%iphotons_PSNR_%.4f_a_%.4f',...
    photon_count,psnr_raw,a),'.png'])


%%
tau_2d=1e-3;
miniscope2d_denoise = tv2d_aniso_haar(b_noise_miniscope, tau_2d);
for i=1:70
    miniscope2d_denoise = tv2d_aniso_haar(miniscope2d_denoise, tau_2d);
end
%miniscope2d_denoise = tvdenoise(b_noise_miniscope,1e-9,4);
imagesc(miniscope2d_denoise)
calc_psnr(res_target,miniscope2d_denoise/max(miniscope2d_denoise(:)))
%%
test=(squeeze(max(vol,[],3)));
test=test/max(test(:));
test(test<0)=0;
test=double(test)*255;
test=uint8(test);
rgbImage = ind2rgb(test, parula(256));
rgbImage=rgbImage(:,100:500,:);

%%
pars.use_gpu = 1;
pars.MAXITER = 500;
miniscope_denoised=TV2DFista(b_noise_miniscope,(0.1),0,1000,pars);