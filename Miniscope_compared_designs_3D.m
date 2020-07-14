%Load impulse response stack, h
design_cells = {'random_multifocal','uni','regular','optimized'}
psf_folder = 'D:\Randoscope\dataforrebuttal\newpsf\';
dtstamp = datestr(datetime('now'),'YYYYmmDD_hhMMss');
savepath = ['D:\\Randoscope\\dataforrebuttal\\newpsf\\3D_recons\\',dtstamp,'\\'];
mkdir(savepath)
test_obj_file = 'D:\Randoscope\dataforrebuttal\newpsf\test_volumes\pt_grid_512x512x72.mat';
h1 = figure(1);clf
options.fighandle = h1;
options.stepsize = 2e-2;
options.convTol = 8.2e-12;

options.maxIter = 3000;

options.residTol = .2;
options.momentum = 'nesterov';
options.disp_figs = 1;
options.disp_fig_interval = 100;   %display image this often
options.print_interval = 50;
boundary_condition = 'pad';
nocrop = @(x)x;
options.disp_crop = @(x)gather(real(sum(x,3)));
options.disp_gamma = 1/2.2;
options.known_input = 0;
options.force_real = 1;
init_style = 'zero';
Nz = 72;
%%
tau_list = logspace(-4,-1.5,10);
reg_list = {'tv'};
for regidx = 1:numel(reg_list)
    for idx = 1:numel(design_cells)
        design = design_cells{idx};
        regularizer = reg_list{regidx};
        switch lower(design)
            case('optimized')
                psf_file = 'psf_aber_ds.mat';
                h_in = load([psf_folder,psf_file]);
                ht = permute(h_in.psf_aber_ds,[2 3 1]);
            case('random_multifocal')
                psf_file = 'psf_multi_ds.mat';
                h_in = load([psf_folder,psf_file]);
                ht = permute(h_in.psf_noaber_multi_mid_ds,[2 3 1]);
            case('uni')
                psf_file='psf_uni_ds.mat';
                h_in = load([psf_folder,psf_file]);
                ht = permute(h_in.psf_noaber_uni_mid_ds,[2 3 1]);
            case('regular')
                psf_file = 'psf_reg_ds.mat'
                h_in = load([psf_folder,psf_file]);
                ht = permute(h_in.psf_noaber_reg_mid_ds,[2 3 1]);
        end
        
        
        ds = 1;
        
        h = zeros(size(ht,1)*ds,size(ht,2)*ds,Nz);
        for m = 1:Nz
            h(:,:,m) = imresize(ht(:,:,m),ds,'box');
        end
        clear ht;
        
        h = single(h);
        hn = sum(sum(abs(h(:,:,1))));
        h = h./hn;
        options.xsize = size(h);
        %z = h_in.z;
        clear h_in
        
        
        %define problem size
        NX = size(h,2);
        NY = size(h,2);
        NZ = size(h,3);
        
        %define crop and pad operators to handle 2D fft convolution
        switch lower(boundary_condition)
            case('circular')
                pad = @(x)x;
                crop = @(x)x;
            case('pad')
                pad = @(x)padarray(x,[size(h,1)/2,size(h,2)/2],0,'both');
                cc = gpuArray(single(size(h,2)/2+1):(3*size(h,2)/2));
                rc = gpuArray(single(size(h,1)/2+1):(3*size(h,1)/2));
                crop = @(x)x(rc,cc,:);
        end
        
        ifftshift2 = @(x)ifftshift(ifftshift(x,1),2);
        fftshift2 = @(x)fftshift(fftshift(x,1),2);
        H = fft2(ifftshift2(gpuArray(single(pad(h)))));
        Hconj = conj(H);
        
        % H_ms = fft2(ifftshift(psf_ms));
        % blur_ms = @(x)ifft2(fft2(x).*H_ms);
        % Define function handle for forward A(x)
        %A3d = @(x)A_lensless_3d(h,x,pad,crop,1);
        A3d = @(x)crop(real(ifft2(sum(fft2(pad(x)).*H,3))));
        % Define handle for A adjoint
        Aadj_3d = @(x)A_adj_lensless_3d_v3(Hconj,x,crop,pad);
        
        
        % Make or load sensor measurement
        meas_type = 'simulated';
        photon_count = 1000;
        switch lower(meas_type)
            case 'simulated'
                obj_in = load(test_obj_file);
                %obj_in = load('D:\Randoscope\dataforrebuttal\zebrafish_resampled_512x512x60.mat');
                obj = obj_in.stackout;
                obj = obj/max(obj(:));
                obj = obj*photon_count;
                sp = 5;
                pct = prctile(obj(:),100-sp);
                obj = obj .* (obj>pct);
                b = single(imnoise(double(A3d(obj))*1e-12,'poisson')*1e12);
                
            case 'measured'
                im_dir = 'C:\Users\herbtipa\Dropbox\3D data\';
                imfname = 'usaf_tilt_reverse.png';
                bin = double(imread([im_dir,imfname]));
                b = gpuArray(imresize(bin,1/2*ds,'box'));
        end
        %%
        % Define gradient handle
        GradErrHandle = @(x)linear_gradient_b(x,A3d,Aadj_3d,b);
        
        % Prox handle
        
        psnr_best = 0;
        ssim_best = 0;
        ssim_list = 0./zeros(size(tau_list));
        psnr_list = 0./zeros(size(tau_list));
        h2 = figure(2);
        clf
        h3 = figure(3);
        clf
        for ii = 1:length(tau_list)
            tau = gpuArray(single(tau_list(ii)));
            %prox_handle = @(x)soft_nonneg(x,tau);
            norm1 = @(x)sum(sum(sum(abs(x))));
            
            switch lower(regularizer)
                case('ell_1')
                    prox_handle = @(x)deal(max(x-tau,0), tau*norm1(x));
                case('tv')
                    prox_handle = @(x)deal(1/2*(max(x,0) + tv3d_iso_Haar(x, tau, 1)), tau*norm1(x));
            end
            
            switch lower(init_style)
                case('zero')
                    [xhat, funvals] = proxMin(GradErrHandle,prox_handle,...
                        gpuArray(zeros(NY,NX,NZ,'like',real(H))),...
                        b,options);
                case('xhat')
                    [xhat, funvals] = proxMin(GradErrHandle,prox_handle,xhat,b,options);
                case('atb')
                    [xhat, funvals] = proxMin(GradErrHandle,prox_handle,Aadj_3d(b),b,options);
            end
            psnr_list(ii) = psnr(double(gather(xhat))/photon_count,obj/photon_count,1);
            ssim_list(ii) = ssim(double(gather(xhat))/photon_count,obj/photon_count);
            if (psnr_list(ii)>psnr_best)
                improvement = true;
                psnr_best = psnr_list(ii);
                xhat_best = gather(xhat);
                tau_best = gather(tau);
            end
            
            if ssim_list(ii)>ssim_best
                improvement = true;
                ssim_best = ssim_list(ii);
                xhat_best_ssim = gather(xhat);
                tau_best_ssim = gather(tau);
            end
            
            if improvement
                save([savepath,'points_',design,'_photons_',num2str(photon_count),...
                    '_PSNR_',num2str(psnr_best),'_ssim_',num2str(ssim_list(ii)),'_',regularizer,'.mat'],'xhat_best','tau_best',...
                    'tau_list','psnr_list','psnr_best','psf_file','b','test_obj_file',...
                    'prox_handle','regularizer','ssim_list','xhat_best_ssim','tau_best_ssim',...
                    'boundary_condition')
                improvement = false;
            end
            
            set(0,'CurrentFigure',h2)
            clf
            subplot(1,3,1)
            imagesc(squeeze(max(xhat_best,[],1)))
            title(['best ',design,', ',num2str(psnr_best,'%.2f'),' dB'])
            axis image
            
            subplot(1,3,2)
            imagesc(squeeze(max(xhat,[],1)))
            title(['current ',design,', ',num2str(psnr_list(ii),'%.2f'),' dB'])
            axis image
            
            
            subplot(1,3,3)
            plot(tau_list,psnr_list)
            xlabel('tau')
            ylabel('psnr')
            drawnow
            
            set(0,'CurrentFigure',h3)
            clf
            subplot(1,3,1)
            imagesc(squeeze(max(xhat_best_ssim,[],1)))
            title(['best ',design,', ',num2str(ssim_best,'%.2f'),' dB'])
            axis image
            
            subplot(1,3,2)
            imagesc(squeeze(max(xhat,[],1)))
            title(['current ',design,', ',num2str(ssim_list(ii),'%.2f'),' dB'])
            axis image
            
            
            subplot(1,3,3)
            plot(tau_list,ssim_list)
            xlabel('tau')
            ylabel('psnr')
            drawnow
        end
        
        
        
        
        
        
    end
end
