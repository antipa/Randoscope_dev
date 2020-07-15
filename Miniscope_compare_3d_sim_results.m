%gt = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\zebrafish_gt_5pct_512x512x60')
test_dir = 'D:\Randoscope\dataforrebuttal\newpsf\3D_recons\20200714_165924_axial_usaf\'

uni_file = uigetfile([test_dir,'*uni*'],'select unifocal');
reg_file = uigetfile([test_dir,'*regular*'],'select regular');
rand_file = uigetfile([test_dir,'*random*'],'select random multifocal');
opt_file = uigetfile([test_dir,'*opt*'],'select optimized');
%%
xhat_opt = load([test_dir,opt_file])
xhat_reg = load([test_dir,reg_file])
xhat_uni = load([test_dir,uni_file])
xhat_rand = load([test_dir,rand_file]);
gt = load('D:\Randoscope\dataforrebuttal\newpsf\test_volumes\axial_usaf_512x512x72.mat')
%xhat_uni = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\July_10_2020\zebrafish_uni_photons_1000_PSNR_28.516.mat');
%xhat_opt = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\July_10_2020\zebrafish_optimized_photons_1000_PSNR_29.1012.mat');
%xhat_rand = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\July_10_2020\zebrafish_random_multifocal_photons_1000_PSNR_29.0519.mat');
%xhat_opt = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\20200713_192525\points_optimized_photons_1000_PSNR_41.0593_ell_1.mat');
%xhat_reg = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\20200713_192525\points_regular_photons_1000_PSNR_42.629_ell_1.mat');
%xhat_reg = load('D:\Randoscope\dataforrebuttal\newpsf\3D_recons\July_10_2020\zebrafish_regular_photons_1000_PSNR_27.8032.mat');
%%
opt = {0, [], []}
subplot = @(m,n,p) subtightplot(m,n,p,opt{:});

gt_obj = gt.stackout*1000;
figure(1)
projax = 3;
daspect = [1 1 1]
slice = 20;
%plot_func = @(x)imagesc(x(:,:,slice));
plotcrop = @(x)(x(:,100:512-100));
%plot_func = @(x)imagesc(plotcrop(rot90(squeeze(max(x,[],projax)),0)))
plot_func = @(x)imagesc(plotcrop(rot90(squeeze(x(256,:,:)),1)))
clf
subplot(2,3,1)
plot_func(xhat_uni.xhat_best);  %imagesc(squeeze(max(xhat_uni.xhat_best,[],projax)))
axis image
a = gca;
a.DataAspectRatio = daspect;
caxis([0 1000])
title(['unifocal random',num2str(xhat_uni.psnr_best)])
axis off

subplot(2,3,2)
plot_func(xhat_reg.xhat_best);
axis image
a = gca;
a.DataAspectRatio = daspect
caxis([0 1000])
title(['regular ',num2str(xhat_uni.psnr_best)])
axis off

subplot(2,3,5)
plot_func(xhat_rand.xhat_best)
axis image
a = gca;
a.DataAspectRatio = daspect
caxis([0 1000])
title(['bad random multifocal ',num2str(xhat_rand.psnr_best)])
axis off

subplot(2,3,4)
plot_func(xhat_opt.xhat_best)
axis image
a = gca;
a.DataAspectRatio = daspect
caxis([0 1000])
title(['optimized ',num2str(xhat_opt.psnr_best)])
axis off

subplot(2,3,6)

plot_func(gt_obj)
axis image
a = gca;
a.DataAspectRatio = daspect
caxis([0 1000])
title('gt')
axis off

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
for n = 1:72
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
zstart = 1;
xax = (zstart:72)*dz;
figure(20)
clf
plot(xax,psnr_reg_vec(zstart:end))
hold on
plot(xax,psnr_opt_vec(zstart:end))
%plot(xax,psnr_uni_vec(zstart:end))
plot(xax,psnr_rand_vec(zstart:end))
legend('reg','opt','worst rand')
ylabel('psnr dB')
xlabel('z \mu m')
title('slice PSNR vs depth')
%%
psnr(double(xhat_reg.xhat_best(:,:,zstart:end)),gt_obj(:,:,zstart:end),1000)
psnr(double(xhat_opt.xhat_best(:,:,zstart:end)),gt_obj(:,:,zstart:end),1000)
psnr(double(xhat_uni.xhat_best(:,:,zstart:end)),gt_obj(:,:,zstart:end),1000)
psnr(double(xhat_rand.xhat_best(:,:,zstart:end)),gt_obj(:,:,zstart:end),1000)

%%
dtstamp = datestr(datetime('now'),'YYYYmmDD_hhMMss');
px = 4.541;  %Pixel size in microns/pixel in sensor space
Mag = 5.2;   %Magnification
px_obj = px/Mag;   %Object space microns/pixel
dz = 5;   %microns

pmax = 1200;

design_list = {'reg','worst_rand','opt','uni'};

for n = 1:numel(design_list)
    filebase = [test_dir,design_list{n}];
    slicefile = [filebase,'_xz_slice.png']
    
end

slice_op = @(x)squeeze(x(256,:,:))
xout_reg.slice = slice_op(xhat_reg.xhat_best);
imagesc(xout_reg.slice)
caxis([0 pmax])