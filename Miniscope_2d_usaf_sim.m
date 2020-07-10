usaf_in = mean(double(imread('/Users/nick.antipa/Documents/USAF-1951_gps4-7_pixel_size_132nm.png')),3);
px_in = 0.132;   %Micron/px

px_target = .76;   %Micron/px
ds = px_in/px_target;
usaf_origin_px = [3700,4100];
usaf_origin_um = usaf_origin_px * px_in;   %row,col not x,y

FoV = [390, 390];

usaf_rows = [round((usaf_origin_um(1)-FoV(1)/2)/px_in) , round((usaf_origin_um(1)+FoV(1)/2)/px_in)];
usaf_rows = min(max(usaf_rows,1),size(usaf_in,1));
usaf_cols = [round((usaf_origin_um(2)-FoV(2)/2)/px_in) , round((usaf_origin_um(2)+FoV(2)/2)/px_in)];
usaf_cols = min(max(usaf_cols,1),size(usaf_in,2));

usaf_crop = imresize(1-usaf_in(usaf_rows(1):usaf_rows(2),usaf_cols(1):usaf_cols(2))/255,ds,'box');
[Ny,Nx] = size(usaf_crop);
if mod(Ny,2) == 0
    usaf_crop = usaf_crop(1:end-1,:);
end
if mod(Nx,2) == 0
    usaf_crop = usaf_crop(:,1:end-1);
end
[Ny,Nx] = size(usaf_crop);
photon_count = 100;
read_noise_std = .0126;


x_ms = [0:(Nx-1)]*px_target;
x_ms = x_ms - mean(x_ms);
y_ms = [0:(Ny-1)]*px_target;
y_ms = y_ms - mean(y_ms);
[X_ms, Y_ms] = meshgrid(x_ms,y_ms);
fwhm = 2.2;
sigma = fwhm/2.355;
psf_ms = 1/sigma/sqrt(2*pi)*exp(- (X_ms.^2 + Y_ms.^2)/2/sigma^2);
psf_ms = psf_ms/sum(sum(psf_ms));

imagesc(psf_ms)


H_ms = fft2(ifftshift(psf_ms));
blur_ms = @(x)ifft2(fft2(x).*H_ms);
usaf_blur = blur_ms(usaf_crop);


im_noisy = imnoise(photon_count*usaf_blur*1e-12,'poisson')*1e12/photon_count;
im_noisy = max(im_noisy + randn(size(im_noisy))*read_noise_std,0);

imagesc(im_noisy)
axis image

ntau = 20;
tau_ms_list = linspace(.003,.2,ntau);
psnr_list_ms = zeros(1,ntau);
n = 0;
figure(1)
clf
psnr_best = 0;
%%
for n = 1:ntau
    tau_ms = tau_ms_list(n);
    psrs.epsilon = 1e-12;
    pars.MAXITER = 500;
    pars.use_gpu = 0;
    [usaf_denoised, iters, fun_all] =  TV2DFista(im_noisy,tau_ms,0,10,pars);
    iters
    imagesc(usaf_denoised);
    axis image
    psnr_list_ms(n) = psnr(usaf_denoised,usaf_crop,1);
    if psnr_list_ms(n) > psnr_best
        recon_best = usaf_denoised;
        index_best = n;
        tau_best = tau_ms;
        psnr_best = psnr_list_ms(n);
    end
    title(sprintf('PSNR: %.2f dB',psnr_list_ms(n)))
    drawnow
end


%%
clf
usaf_wrong_norm = usaf_denoised/max(usaf_denoised(:))*photon_count;
plot(usaf_wrong_norm(1,:))
hold on
plot(usaf_crop(1,:)*photon_count)
plot(usaf_denoised(1,:))

legend('noisy max scale','gt','noisy right scale')
hold off