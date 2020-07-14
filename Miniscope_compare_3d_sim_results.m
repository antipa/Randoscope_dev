%gt = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\zebrafish_gt_5pct_512x512x60')
gt = load('D:\Randoscope\dataforrebuttal\newpsf\test_volumes\pt_grid_512x512x72.mat');
xhat_uni = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\July_10_2020\zebrafish_uni_photons_1000_PSNR_28.516.mat');
%xhat_opt = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\July_10_2020\zebrafish_optimized_photons_1000_PSNR_29.1012.mat');
xhat_rand = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\July_10_2020\zebrafish_random_multifocal_photons_1000_PSNR_29.0519.mat');
xhat_opt = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\20200713_192525\points_optimized_photons_1000_PSNR_41.0593_ell_1.mat');
xhat_reg = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\20200713_192525\points_regular_photons_1000_PSNR_42.629_ell_1.mat');
%xhat_reg = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\July_10_2020\zebrafish_regular_photons_1000_PSNR_27.8032.mat');
%%
gt_obj = gt.stackout*1000;
figure(1)
projax = 3;
daspect = [1 1 1]
slice = 20;
%plot_func = @(x)imagesc(x(:,:,slice));
plot_func = @(x)imagesc(squeeze(max(x,[],projax)))
clf
% subplot(2,3,1)
% plot_func(xhat_uni.xhat_best);  %imagesc(squeeze(max(xhat_uni.xhat_best,[],projax)))
% axis image
% a = gca;
% a.DataAspectRatio = daspect;
% caxis([0 1000])

subplot(1,3,1)
plot_func(xhat_reg.xhat_best);
axis image
a = gca;
a.DataAspectRatio = daspect
caxis([0 1000])
title('regular')

% subplot(2,3,3)
% plot_func(xhat_rand.xhat_best)
% axis image
% a = gca;
% a.DataAspectRatio = daspect
% caxis([0 1000])

subplot(1,3,2)
plot_func(xhat_opt.xhat_best)
axis image
a = gca;
a.DataAspectRatio = daspect
caxis([0 1000])
title('optimized')

subplot(1,3,3)

plot_func(gt_obj)
axis image
a = gca;
a.DataAspectRatio = daspect
caxis([0 1000])
title('gt')

% subplot(2,3,6)
% 
% plot_func(xhat_opt.xhat_best - gt_obj)
% axis image
% a = gca;
% a.DataAspectRatio = daspect
% caxis([0 1000])

%%
psnr_reg_vec = []
psnr_opt_vec = []
psnr_uni_vec = []
psnr_rand_vec = []
for n = 1:60
    psnr_reg_vec(n) = psnr(double(xhat_reg.xhat_best(:,:,n)),...
        gt_obj(:,:,n),1000);
    psnr_opt_vec(n) = psnr(double(xhat_opt.xhat_best(:,:,n)),...
        gt_obj(:,:,n),1000);
    psnr_uni_vec(n) = psnr(double(xhat_uni.xhat_best(:,:,n)),...
        gt_obj(:,:,n),1000);
    psnr_rand_vec(n) = psnr(double(xhat_rand.xhat_best(:,:,n)),...
        gt_obj(:,:,n),1000);
end
dz = 5;
zstart = 20;
xax = (zstart:60)*dz;
figure(20)
clf
plot(xax,psnr_reg_vec(zstart:end))
hold on
plot(xax,psnr_opt_vec(zstart:end))
plot(xax,psnr_uni_vec(zstart:end))
plot(xax,psnr_rand_vec(zstart:end))
legend('reg','opt','uni','rand')
ylabel('psnr dB')
xlabel('z \mu m')
title('slice PSNR vs depth')
%%
psnr(double(xhat_reg.xhat_best(:,:,zstart:end)),gt_obj(:,:,zstart:end),1000)
psnr(double(xhat_opt.xhat_best(:,:,zstart:end)),gt_obj(:,:,zstart:end),1000)
psnr(double(xhat_uni.xhat_best(:,:,zstart:end)),gt_obj(:,:,zstart:end),1000)
psnr(double(xhat_rand.xhat_best(:,:,zstart:end)),gt_obj(:,:,zstart:end),1000)