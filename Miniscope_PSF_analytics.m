%%
pth = 'D:\Randoscope\dataforrebuttal\PSFs';
in = load([pth,'\psf_uni_middle.mat']);
psf_stack_uni = permute((in.psf_noaber_uni_mid_ds), [2 3 1]);


in = load([pth,'\psf_multi_middle.mat']);
%psf_stack_rando = permute(in.psf_montebest_ds, [2 3 1]);
psf_stack_rando = permute((in.psf_noaber_multi_mid_ds), [2 3 1]);


in = load([pth,'\psf_regular_middle.mat']);
psf_stack = permute((in.psf_noaber_reg_mid_ds), [2 3 1]);
h6 = figure(6),clf
h1 = figure(1),clf
h5 = figure(5),clf
Ascore_reg = []
Ascore_rand = []
Ascore_uni = []
ripple_reg = []
ripple_rando = []
ripple_uni = []
% Designed for 3.5 micron resolution object space
% Convert to lp/micron
% fc = 2*NA/lambda
% Rayleigh = 3.5 = .61 lambda/NA

px = 7.3;  %Pixel size in microns/pixel in sensor space
Mag = 5.2;   %Magnification
px_obj = px/Mag;   %Object space microns/pixel
[Ny,Nx] = size(psf_rando);
R = 3.5;
lambda = 0.51;
NA = .61*lambda/R;
fc_adjust = 1/2;
fc = 2*NA/lambda*fc_adjust;   %Cycles per micron
fmax = 1/2/px_obj;
fgrid_rad = (0:nbins-1)/nbins*fmax;
df_rad = mean(diff(fgrid_rad));
fc_rad_px = fc/df_rad;
fgrid_lin = (0:Ny/2-1)/Ny*2*fmax;
fc_grid_px = fc/mean(diff(fgrid_lin));

[X,Y] = meshgrid(-255.5:1:255.5,-255.5:1:255.5);
c = sqrt(X.^2+Y.^2)<=(fc_grid_px+3);
fftcorr = @(x,y)gather(real(ifft2(fft2(ifftshift(x)).*conj(fft2(y)))));
c2 = sqrt(X.^2+Y.^2)<=fc_grid_px/2;
c2 = fftcorr(c2,c2);

c2 = gpuArray(single(c2/max(max(c2))).^2);
C2 = ifft2(c2);
ripple_err = @(x,c)gather(norm(x - c/max(c(:))*max(x(:)),'fro'));
Nz =72;
Astar = @(x)c.*(1./(abs(x)+.00000001));
[X,Y] = meshgrid(1:512,1:512);
mmm = X>240 & X<275 & Y<255 & Y>230;
for zplane = 40
    %zplane = 24
    
    psf_uni=psf_stack_uni(:,:,zplane);%6use 20 and 60
    psf_uni = psf_uni/sum(sum(psf_uni));
    
    psf_rando=psf_stack_rando(:,:,zplane);% use 20 and 60, end-5 for noise sims
    psf_rando = psf_rando/sum(sum(psf_rando));
    
    regoffset = 0;
    psf_regular = psf_stack(:,:,zplane+regoffset);  %use 72 and 32
    psf_regular = psf_regular/sum(sum(psf_regular));
    
    %This plots spectrum of one point
    psfMask = psf_uni .* mmm;
    spectMask = fftshift(abs(fft2(ifftshift(psfMask))));
    [spectrad, bns] = radialavg(spectMask,nbins);
    set(0,'CurrentFigure',h1)
    clf
    subplot(1,3,1)
    imagesc(psf_regular)
    axis image
    caxis([0 .002]);
    title('reg')
    
    subplot(1,3,2)
%     imagesc(psfMask)
%     axis image
%     caxis([0 .002]);
    plot(bns*fmax,spectrad)
    hold on
    line([fc fc],[.05 .05])
    title('uni')
    hold off
    subplot(1,3,3)
    imagesc(psf_rando)
    axis image
    caxis([0 .002]);
    title('rando')
    
    pad2d = @(x)padarray(x,[size(x,1)/2,size(x,2)/2],'both');
    
    crop2d = @(x)x(size(x,1)/4+1:3*size(x,1)/4, size(x,2)/4+1:3*size(x,2)/4);
    
    %
    % psfacorr = crop2d(xcorr2(psf_regular,psf_regular));
    % psfacorr_rando = crop2d(xcorr2(psf_rando,psf_rando));
    % psfacorr_uni = crop2d(xcorr2(psf_uni,psf_uni));
    
    psfspect = gather(fftshift(abs(fft2(psf_regular)).^2));
    psfspect_rando = gather(fftshift(abs(fft2(psf_rando)).^2));
    psfspect_uni = gather(fftshift(abs(fft2(psf_uni)).^2));
    
    ripple_rando(zplane) = ripple_err(psf_rando,c2);
    ripple_reg(zplane) = ripple_err(psf_regular,c2);
    ripple_uni(zplane) = ripple_err(psf_uni,c2);
    
    nbins = 200;
    [psavg, avgbins] = radialavg(abs(psfspect),nbins);
    [psavg_rando, avgbins] = radialavg(abs(psfspect_rando),nbins);
    [psavg_uni,avgbins] = radialavg(abs(psfspect_uni),nbins);
    set(0,'CurrentFigure',h5);
    clf
    plot(avgbins,psavg)
    hold on
    plot(avgbins,psavg_rando)
    plot(avgbins,psavg_uni)
    legend('reg','designed','uni')
    axis([0 .5 0 .1])
    hold off
    
    
    
    
    
    Astar_reg_im =Astar(psfspect);
    Astar_rand_im = Astar(psfspect_rando);
    Astar_uni_im = Astar(psfspect_uni);
    Ascore_reg(zplane) = sum(sum(Astar_reg_im));
    Ascore_rand(zplane) = sum(sum(Astar_rand_im));
    Ascore_uni(zplane) = sum(sum(Astar_uni_im));
    [Astar_reg, avgbins] = radialavg(Astar_reg_im,nbins);
    [Astar_rand, avgbins] = radialavg(Astar_rand_im,nbins);
    [Astar_uni, ~] = radialavg(Astar_uni_im, nbins);
    acorr_reg = real(ifft2(ifftshift(psfspect)));
    acorr_uni = real(ifft2(ifftshift(psfspect_uni)));
    acorr_rand = real(ifft2(ifftshift(psfspect_rando)));
    if zplane == 1
        Astar_reg_mat = Astar_reg;
        Astar_uni_mat = Astar_uni;
        Astar_rand_mat = Astar_rand;
        acorrslice_reg = acorr_reg(1,:);
        acorrslice_uni = acorr_uni(1,:);
        acorrslice_rand = acorr_rand(1,:);
    else
    
        Astar_reg_mat = cat(1,Astar_reg_mat,Astar_reg);
        Astar_uni_mat = cat(1,Astar_uni_mat,Astar_uni);
        Astar_rand_mat = cat(1,Astar_rand_mat,Astar_rand);
        acorrslice_reg = cat(1,acorrslice_reg,acorr_reg(1,:));
        acorrslice_uni = cat(1,acorrslice_uni,acorr_uni(1,:));
        acorrslice_rand = cat(1,acorrslice_rand,acorr_rand(1,:));
        
    end
    
    set(0,'CurrentFigure',h6)
    semilogy(fgrid_rad,Astar_reg)
    hold on
    semilogy(fgrid_rad,Astar_rand)
    semilogy(fgrid_rad,Astar_uni)
    line([fc, fc],[1 1e12])
    legend(sprintf('reg %.2g',Ascore_reg(zplane)),...
        sprintf('opt %.2g',Ascore_rand(zplane)),...
        sprintf('uni %.2g',Ascore_uni(zplane)))
    hold off
    drawnow
end


%%

zvec = (0:71) * 5;
sm = 1;
filt_kern = 1/sm*ones(1,sm);
Ascore_reg_sm = filter(filt_kern,1,medfilt1(Ascore_reg,1));
Ascore_uni_sm = filter(filt_kern,1,medfilt1(Ascore_uni,1));
Ascore_rand_sm = filter(filt_kern,1,medfilt1(Ascore_rand,1));

figure(7),clf

plot(zvec,Ascore_reg_sm,'k-.','LineWidth',2)
hold on
plot(zvec,Ascore_uni_sm,'k','LineWidth',2)
plot(zvec,Ascore_rand_sm,'r','LineWidth',2)
% a = gca
% a.XTick = linspace(0
legend('Regular unifocal','irregular unifocal','irregular multifocal (ours)')
title('sum(1/|MTF|^2)')
xlabel('depth \mu m')
xlim([0 360-5])
ylim([0 max(Ascore_uni_sm(:))])
ylabel('sum(1/|MTF|)')
hold off
%%

figure(8),clf

plot(ripple_reg,'k-.','LineWidth',2)
hold on
plot(ripple_uni,'k','LineWidth',2)
plot(ripple_rando,'r','LineWidth',2)

legend('Regular unifocal','irregular unifocal','irregular multifocal (ours)')
title('frequency space error')
hold off
mxc = 14;
sc = 1;
px_xc = px_obj*1/sc;  %microns/pixel of scaled image
prep_acorr = @(x)imresize(x(:,1:mxc)./max(x,[],2),sc,'bicubic');

figure(9)
clf

subplot(1,3,1)
imagesc(prep_acorr(acorrslice_reg))
ylabel('depth')
axis image
title('reg')
subplot(1,3,2)
imagesc(prep_acorr(acorrslice_uni))
ylabel('depth')
axis image
title('uni')

subplot(1,3,3)
imagesc(prep_acorr(acorrslice_rand))
ylabel('depth')
axis image
title('designed')

colormap jet

temp = prep_acorr(acorrslice_reg);
 
yplot = ((1:size(temp,2))-1)*px_xc*2  %The *2 is to make it FWHM
xplot = ((1:size(temp,1))-1)*72*5/size(temp,1);
[Xplt,Yplt] = meshgrid(xplot,yplot);
fwhm_contour = @(x,pct,clr)contour(Xplt,Yplt,rot90(x,-1),[pct pct],clr)
figure(10)
clf
fwhm_contour((prep_acorr(acorrslice_reg)),0.48,'k');
hold on
fwhm_contour((prep_acorr(acorrslice_uni)),0.48,'r');
fwhm_contour((prep_acorr(acorrslice_rand)),0.48,'b');

title('FWHM vs z')

% This stuff does it by counting pixels:
% acnorm_reg = prep_acorr(acorrslice_reg);
% acnorm_uni = prep_acorr(acorrslice_uni);
% acnorm_rand = prep_acorr(acorrslice_rand);
% make_fwhm = @(x,pct)sum(x>=pct,2);
% 
% 
% 
% 
% 
% fwhm_uni =make_fwhm(acnorm_uni,0.5)*px_obj*2;
% fwhm_reg = make_fwhm(acnorm_rand,0.5)*px_obj*2;
% plot(xplot,flipud(fwhm_uni))
% plot(xplot,flipud(fwhm_reg))
legend('Regular','unifocal','designed')
hold off 

%%

% 
% plot(ripple_reg,'k-.','LineWidth',2)
% hold on
% plot(ripple_uni,'k','LineWidth',2)
% plot(ripple_rando,'r','LineWidth',2)
% 
% legend('Regular unifocal','irregular unifocal','irregular multifocal (ours)')

zout = 1;
zin = 40
figure(13)
clf

a1 = semilogy(fgrid_rad,Astar_reg_mat(zout,:),'k-.','LineWidth',2)
hold on
a2 = semilogy(fgrid_rad,Astar_uni_mat(zout,:),'k','LineWidth',2)
a3 = semilogy(fgrid_rad,Astar_rand_mat(zout,:),'r','LineWidth',2)
%line([fc, fc],[1 1e12])
axis([0 fc 10 max(Astar_reg_mat(zout,:))])
legend('Regular unifocal','irregular unifocal','irregular multifocal (ours)','Location','southeast')
title('z=0 (-200 \mu m out of focus)')
hold off

figure(14)
clf
semilogy(fgrid_rad,Astar_reg_mat(zin,:),'k-.','LineWidth',2)
hold on
semilogy(fgrid_rad,Astar_uni_mat(zin,:),'k','LineWidth',2)
semilogy(fgrid_rad,Astar_rand_mat(zin,:),'r','LineWidth',2)
axis([0 fc 10 max(Astar_reg_mat(zout,:))])
legend('Regular unifocal','irregular unifocal','irregular multifocal (ours)','Location','southeast')
title('z=200 \mu m (In focus)')

hold off

% Same thing but linear scale
figure(15)
clf

a1 = plot(fgrid_rad,Astar_reg_mat(zout,:),'k-.','LineWidth',2)
hold on
a2 = plot(fgrid_rad,Astar_uni_mat(zout,:),'k','LineWidth',2)
a3 = plot(fgrid_rad,Astar_rand_mat(zout,:),'r','LineWidth',2)
%line([fc, fc],[1 1e12])
axis([0 fc 10 max(Astar_reg_mat(zout,:))])
legend('Regular unifocal','irregular unifocal','irregular multifocal (ours)','Location','northwest')
title('z=0 (-200 \mu m out of focus)')
hold off

figure(16)
clf
plot(fgrid_rad,Astar_reg_mat(zin,:),'k-.','LineWidth',2)
hold on
plot(fgrid_rad,Astar_uni_mat(zin,:),'k','LineWidth',2)
plot(fgrid_rad,Astar_rand_mat(zin,:),'r','LineWidth',2)
axis([0 fc 10 max(Astar_reg_mat(zout,:))])
legend('Regular unifocal','irregular unifocal','irregular multifocal (ours)','Location','northwest')
title('z=200 \mu m (In focus)')

hold off
%%
outpth = 'D:\\Randoscope\\dataforrebuttal\\PSFs'
vec = @(x)x(:);
imnorm = max([max(vec(psf_stack(:,:,zin))) max(vec(psf_stack_uni(:,:,zin))) max(vec(psf_stack_rando(:,:,zin)))])
cmap = double(imread(['D:\Randoscope\dataforrebuttal\cmap_imagej_fire.tif']))/255
ncolors = length(colormap(cmap));
r1 = 80;
r2 = 512-100;
c1 = 90;
c2 = 512-90;
make_outim = @(im,cmap,nm)ind2rgb(gray2ind(uint8(im(r1:r2,c1:c2)/nm*255),length(colormap(cmap))),colormap(cmap));
rando_in = make_outim(psf_stack_rando(:,:,zin),cmap,imnorm);
reg_in = make_outim(psf_stack(:,:,zin),cmap,imnorm);
uni_in = make_outim(psf_stack_uni(:,:,zin),cmap,imnorm);
reg_out = make_outim(psf_stack(:,:,zout),cmap,imnorm);
uni_out = make_outim(psf_stack_uni(:,:,zout),cmap,imnorm);
rando_out = make_outim(psf_stack_rando(:,:,zout),cmap,imnorm);

imwrite(rando_in,[outpth,'\rando_in.png'])
imwrite(reg_in,[outpth,'\regular_in.png'])
imwrite(uni_in,[outpth,'\uni_in.png'])
imwrite(rando_out,[outpth,'\rando_out.png'])
imwrite(reg_out,[outpth,'\reg_out.png'])
imwrite(uni_out,[outpth,'\uni_out.png'])
imshow(uni_out)
%%
surfs_in = load(['D:\Randoscope\dataforrebuttal\PSFs\surfaces_with_apreture.mat'])
optsurf_in = load('D:\Randoscope\dataforrebuttal\PSFs\optimized_mask');
smax = 4e-3;
px_srf = 3600/2400; 
r1 = 512-2*(size(reg_out,1)/2)
r2 = 512+2*(size(reg_out,1)/2)
c1 = r1
c2 = r2
make_outim = @(im,cmap,nm)ind2rgb(gray2ind(uint8(im(r1:r2,c1:c2)/nm*255),length(colormap(cmap))),colormap(cmap));
surfreg = make_outim(imresize(surfs_in.Tnoaber_reg_mid,[1024,1024]),'parula',smax);
surfuni = make_outim(imresize(surfs_in.Tnoaber_uni_mid,[1024,1024]),'parula',smax);
surfrand = make_outim(imresize(surfs_in.Tnoaber_multi_mid,[1024,1024]),'parula',smax);
surfopt = make_outim(imresize(optsurf_in.Taber_multi,[1024,1024]),'parula',smax);
imshow(surfrand)

imwrite(surfreg,[outpth,sprintf('\\regular_surface_%g_um_FoV_cmax_%g_um.png',(r2-r1)*px_srf*2,smax*1e3)])
imwrite(surfuni,[outpth,sprintf('\\unifocal_surface_%g_um_FoV_cmax_%g_um.png',(r2-r1)*px_srf*2,smax*1e3)])
imwrite(surfrand,[outpth,sprintf('\\multifocal_surface_%g_um_FoV_cmax_%g_um.png',(r2-r1)*px_srf*2,smax*1e3)])
imwrite(surfopt,[outpth,sprintf('\\optimized_surface_%g_um_FoV_cmax_%g_um.png',(r2-r1)*px_srf*2,smax*1e3)])


%% Make axial correlation plots for resolution analysis

Nz = size(psf_stack_uni,3);
axmat_uni = zeros(Nz,Nz);
axmat_regular = zeros(Nz,Nz);
axmat_rando = zeros(Nz,Nz);
corrmat_uni = zeros(Nz,Nz);
corrmat_regular =  zeros(Nz,Nz);
corrmat_rando = zeros(Nz,Nz);
%%
dot_prod = @(x,n,m)gather(sum(sum(x(:,:,n).*x(:,:,m))));
max_corr = @(x,n,m)max(max(fftcorr(x(:,:,n),x(:,:,m))));
figure(9),clf
for nn = 1:Nz
    for mm = nn:Nz
        axmat_uni(nn,mm) = dot_prod(psf_stack_uni,nn,mm);
        axmat_regular(nn,mm) = dot_prod(psf_stack,nn,mm);
        axmat_rando(nn,mm) = dot_prod(psf_stack_rando,nn,mm);
        corrmat_uni(nn,mm) = max_corr(psf_stack_uni,nn,mm);
        corrmat_regular(nn,mm) = max_corr(psf_stack,nn,mm);
        corrmat_rando(nn,mm) = max_corr(psf_stack_rando,nn,mm);
    end
    subplot(2,3,1)
    imagesc(axmat_uni)
    axis image
    
    title('uni')
    
    subplot(2,3,2)
    imagesc(axmat_regular)
    axis image
    
    title('regular')
    
    subplot(2,3,3)
    imagesc(axmat_rando)
    axis image
    
    title('rando')
    
    subplot(2,3,4)
    imagesc(corrmat_uni)
    axis image
    
    title('uni maxcorr')
    
    subplot(2,3,5)
    imagesc(corrmat_regular)
    axis image
    
    title('regular maxcorr')
    
    subplot(2,3,6)
    imagesc(corrmat_rando)
    axis image
    
    title('rando maxcorr')
    drawnow
    
end