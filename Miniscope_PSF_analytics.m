%%
in = load('C:\Users\kyrollos\Downloads\Miniscope3D-master\dataforrebuttal\psf_noaber_uni.mat');   
psf_stack_uni = permute(in.psf_noaber_uni_mid_ds, [2 3 1]);
psf_uni=psf_stack_uni(:,:,15);%6use 20 and 60
psf_uni = psf_uni/sum(sum(psf_uni));

in = load('C:\Users\kyrollos\Downloads\Miniscope3D-master\dataforrebuttal\psf_noaber_montebest.mat');   
psf_stack_rando = permute(in.psf_montebest_ds, [2 3 1]);
psf_rando=psf_stack_rando(:,:,15);% use 20 and 60, end-5 for noise sims
psf_rando = psf_rando/sum(sum(psf_rando));

in = load('C:\Users\kyrollos\Downloads\Miniscope3D-master\dataforrebuttal\psf_noaber_reg.mat');
psf_stack = permute(in.psf_noaber_reg_ds, [2 3 1]);
psf_regular = psf_stack(:,:,72);  %use 72 and 32
psf_regular = psf_regular/sum(sum(psf_regular));

figure(1)
clf
subplot(1,3,1)
imagesc(psf_regular)
axis image
caxis([0 .002]);
title('reg')

subplot(1,3,2)
imagesc(psf_uni)
axis image
caxis([0 .002]);
title('uni')

subplot(1,3,3)
imagesc(psf_rando)
axis image
caxis([0 .002]);
title('rando')

psfacorr = crop2d(xcorr2(psf_regular,psf_regular));
psfacorr_rando = crop2d(xcorr2(psf_rando,psf_rando));
psfacorr_uni = crop2d(xcorr2(psf_uni,psf_uni));

psfspect = gather(fftshift(fft2(psfacorr)));
psfspect_rando = gather(fftshift(fft2(psfacorr_rando)));
psfspect_uni = gather(fftshift(fft2(psfacorr_uni)));


nbins = 200;
[psavg, avgbins] = radialavg(abs(psfspect),nbins);
[psavg_rando, avgbins] = radialavg(abs(psfspect_rando),nbins);
[psavg_uni,avgbins] = radialavg(abs(psfspect_uni),nbins);
figure(5)
clf
plot(avgbins,psavg)
hold on
plot(avgbins,psavg_rando)
plot(avgbins,psavg_uni)
legend('reg','designed','uni')
axis([0 .5 0 .1])
hold off




[X,Y] = meshgrid(-255.5:1:255.5,-255.5:1:255.5);
c = sqrt(X.^2+Y.^2)<=100;
Astar = @(x)c.*(1./abs(x));
Astar_reg_im =Astar(psfspect);
Astar_rand_im = Astar(psfspect_rando);
Astar_uni_im = Astar(psfspect_uni);
Ascore_reg = sum(sum(Astar_reg_im));
Ascore_rand = sum(sum(Astar_rand_im));
Ascore_uni = sum(sum(Astar_uni_im));
[Astar_reg, avgbins] = radialavg(Astar_reg_im,nbins);
[Astar_rando, avgbins] = radialavg(Astar_rand_im,nbins);
[Astar_uni, ~] = radialavg(Astar_uni_im, nbins);
figure(6),clf
semilogy(avgbins,Astar_reg)
hold on
semilogy(avgbins,Astar_rando)
semilogy(avgbins,Astar_uni)
legend(sprintf('reg %.2g',Ascore_reg),sprintf('opt %.2g',Ascore_rand),sprintf('uni %.2g',Ascore_uni))
hold off

%% Make axial correlation plots for resolution analysis
Nz = size(psf_stack_uni,3);
axmat_uni = zeros(Nz,Nz);
axmat_regular = zeros(Nz,Nz);
axmat_rando = zeros(Nz,Nz);
corrmat_uni = zeros(Nz,Nz);
corrmat_regular =  zeros(Nz,Nz);
corrmat_rando = zeros(Nz,Nz);
fftcorr = @(x,y)gather(real(ifft2(fft2(ifftshift(x)).*conj(fft2(y)))));
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
    drawnow
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