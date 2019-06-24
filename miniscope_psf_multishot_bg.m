% Using multiple xy translations of a bead, estimate the psf and the
% background
%
% Forward model: yi = crop(Si(pad(ai * h))) + b
% b: background (non shifting)
% h: desired psf
% ai: attenuation to allow for uneven illumination
% yi: measurement of bead shifted by Si
% Note: Si = Si' = inv(Si)
%
%% Step 1: read images as {yi}
ds = 4;
%ystack = read_im_stack('/Users/nick.antipa/Documents/Diffusers/Miniscope/RandoscopeNanoscribe/Shift_Varying/meas',ds);
%data_dir = '/Users/nick.antipa/Documents/Diffusers/Miniscope/RandoscopeNanoscribe/Shift_variance_june2019';
data_dir = '/Users/nick.antipa/Documents/Diffusers/Miniscope/RandoscopeNanoscribe/Shift_variance_june2019/Dense/'
xystackpath = [data_dir,'/shifted_psf_1_2x_3dtiff.tif'];
ystack = double(read_tiff_stack(xystackpath,ds/2));
%%
%bg = read_im_stack('/Users/nick.antipa/Documents/Diffusers/Miniscope/RandoscopeNanoscribe/Shift_Varying/bg',ds);
%bgpath = '/Users/nick.antipa/Documents/Diffusers/Miniscope/RandoscopeNanoscribe/Shift_variance_june2019/bg_shiftedpsf_4_8um_bettertilt_1-1.tif';
bgpath = [data_dir,'/shifted_psf_background.tif']
bg = 1.003*imresize(double(imread(bgpath)),1/ds,'box');
%good_ids = [54:61,73:81,91:100,111:118];
%good_ids = [1:6,8:39]
%good_ids = 1:size(ystack,3);
good_ids = [1:4,47:56,81:88,91:102,108:191,194:229,233:259,274:280];
bad = [1:4,137,188];
good_ids = good_ids(~(ismember(good_ids,bad)));


%%
yi = ystack(:,:,good_ids);
%yi_no_bg = ystack(:,:,good_ids);
sz = size(bg);
%% Step 2: Estimate shifts using cross correlation. This gives us {Si}
icenter = 104;
pad = @(x)padarray(x,[size(ystack,1), size(ystack,2)],'post');
crop = @(x)x(1:size(ystack,1),1:size(ystack,2));
fftcorr = @(x,y)(ifft2(fft2(pad(x)).*conj(fft2(ifftshift(pad(y))))));

ytest = fftcorr(yi(:,:,1),yi(:,:,icenter));
findpeak_id = @(x)find(x == max(x(:)));



%%
yi_bg = yi - bg;
M = size(yi,3);
Si = @(x,si)circshift(x,si);
yi_bg(:,:,icenter) = yi_bg(:,:,icenter)/norm(yi_bg(:,:,icenter),'fro');
clf
[pr, pc] = ind2sub(2*sz,findpeak_id(fftcorr(yi_bg(:,:,icenter),yi_bg(:,:,icenter))));
si = {};
yi_reg = 0*yi;
R = []
for m = 1:M
    %yi_bg(:,:,m) = (yi(:,:,m) - bg, 3, 10);
    yi_bg(:,:,m) = yi_bg(:,:,m)/norm(yi_bg(:,:,m),'fro');
    %yi(:,:,m) = yi(:,:,m)/norm(yi(:,:,m),'fro');
    [r,c] = ind2sub(2*sz,findpeak_id(fftcorr(remove_hot_pixels(yi_bg(:,:,m),3,.0001),yi_bg(:,:,icenter))));
    si{m} = [-(r-pr),-(c-pc)];
    R(m) = sqrt(si{m}(1)^2 + si{m}(2)^2);
    %hold on
    %scatter(c,r)
    yi_reg(:,:,m) = (Si((yi_bg(:,:,m)),si{m}));
%     imagesc(yi_reg(:,:,m))
%     title(R(m))
%     axis image
%     drawnow
    m
    %pause(1/2)
end
R = R/max(R);
R(isnan(R)) = 1;
%bad = [7, 40];
Rweight = reshape(1-R,[1,1,M]);


%% Step 3: with shifts taken care of, use gradient descent to solve
%   argmin ||ai C Si P(h) + b - yi||
%  {ai, h, b}
% P := pad = C' = (crop)'
% 


yi = yi_bg;

Ny = sz(1);
Nx = sz(2);

fx = linspace(-Nx/2,Nx/2,Nx);
fy = linspace(-Ny/2,Ny/2,Ny);
[Fx, Fy] = meshgrid(fx,fy);
W = fftshift(sqrt(Fx.^2 + Fy.^2)<=8);
B = @(x)fftshift(real(ifft2(W.*x)));



%

niter = 300;
rk = zeros(sz(1),sz(2),M);
hk = yi(:,:,icenter);
ak = ones(1,1,M);
ga = zeros(1,1,M);
%bk = bg/norm(bg,'fro');
Bk = (fft2(ifftshift(bg/norm(bg,'fro'))));
tk = 1;
tkp = 1;

hkp = hk;
Bkp = Bk;
akp = ak;
Gh = rk;
l = [];
mu = 1e-3;
hbar = rk;
tau = 3e-5;



for r = 1:niter
    bk = B(Bk);
    for i = 1:M
        hbar(:,:,i) = crop(Si(pad(hk),-si{i}));
        rk(:,:,i) = Rweight(i).*(ak(i)*hbar(:,:,i) + bk - yi(:,:,i));
    end
    gB = W.*fft2(ifftshift(sum(rk,3)));  %Gradient for background
    
    for i = 1:M
        rks = crop(Si(pad(rk(:,:,i)) ,si{i}));
        Gh(:,:,i) = ak(i).*rks;      %Gradient for psf
        ga(:,:,i) = sum(sum(hk.*rks));
    end
    
    l = cat(1,l,norm(rk(:)));
    gh = sum(Gh.*reshape(1-R,[1,1,M]),3);
    
    tkp = (1+sqrt(1+4*tk^2))/2;
    betakp = (tk-1)/tkp;
    

    
    hkp = hk - mu*gh;
    akp = ak - mu*ga;
    Bkp = Bk - mu*gB;
    
    hkp = max(soft(hkp,tau),-1e-5);
    akp = max(akp,0);
    
    dh = hkp - hk;
    dB = Bkp - Bk;
    da = akp - ak;
    
    hkp = hkp + betakp*dh;
    Bkp = Bkp + betakp*dB;
    akp = akp + betakp*da;
    
    tk = tkp;
    Bk = Bkp;
    ak = akp;
    hk = hkp;
    


    %bk = max(bk,0);
    subplot(2,2,1)
    imagesc(hk)
    title('PSF Estimate')
    axis image
    
    drawnow
    subplot(2,2,2)
    
    imagesc(bk)
    title('Background')
    axis image
    
    subplot(2,2,3)
   
    plot(squeeze(ak))
    title('coefficients')
    subplot(2,2,4)
    plot(l)
    title('loss')
    
end

%% Try estimating bg using low rank business
ymat = vec(yi_reg(:,:,1));
for m = 2:M
    ymat = cat(2,ymat,vec(yi_reg(:,:,m)));
    m
end
%%
tic
rnk = 12;    %Rank of bg
[u,s,v] = svds(ymat,rnk);


u_reshaped = reshape(u,[size(ystack(:,:,1)),rnk]);
toc
%%
vt = v';
yi_svd = 0*yi;
for m = 1:M
    yi_accum = zeros(size(ystack(:,:,1)));
    for r = 1:rnk
        yi_accum = yi_accum + u_reshaped(:,:,r) * s(r,r) * vt(r,m);
    end
    yi_svd(:,:,m) = yi_accum;
    m
end

%% upsample vt
[Ny, Nx] = size(yi(:,:,icenter));
alpha = zeros(M,rnk);
for m = 1:M
    for r = 1:rnk
        alpha(m,r) = s(r,r)*vt(r,m);
    end
end
si_mat = reshape(cell2mat(si)',[2,M]);
xq = -Nx/2+1:Nx/2;
yq = -Ny/2+1:Ny/2;
[Xq, Yq] = meshgrid(xq,yq);

beta = zeros([size(Xq)*2,rnk]);
for r = 1:rnk
    %beta(:,:,r) = interp2(si_mat(2,:)', si_mat(1,:)', alpha(:,r)',Xq, Yq);
    interpolant_r = scatteredInterpolant(-si_mat(2,:)', -si_mat(1,:)', alpha(:,r),'natural','nearest');
    beta(:,:,r) = rot90(padarray(interpolant_r(Xq,Yq),[Ny/2,Nx/2],'replicate'),0);
end


%%
nopad = @(x)x;
nocrop = @(x)x;
pad2d = @(x)padarray(x,[Ny/2,Nx/2],'both');
crop2d = @(x)x(Ny/2+1:3*Ny/2,Nx/2+1:3*Nx/2,:);

%%
H = zeros(size(nopad(u_reshaped(:,:,1)),1), size(nopad(u_reshaped(:,:,1)),2), rnk);
for r = 1:rnk
    H(:,:,r) = fft2(ifftshift(nopad(u_reshaped(:,:,r))));
end
H_conj = conj(H);

%%
imd = double(imread([data_dir,'/usaf_dim_2x_slice40_100bias.png']))-100;
imd = imresize(imd/max(imd(:)),2/ds,'box');

beta_crop = crop2d(beta);
A = @(x)A_svd(H, beta_crop, x, nocrop);
%A = @(x)A_svd_3d(x,beta_crop,H);
A_adj = @(x)A_adj_svd(H_conj, beta_crop, x, nopad);
%A_adj = @(x)A_adj_svd_3d(x,beta_crop,H_conj);

GradErrHandle = @(x) linear_gradient(x,A,A_adj,imd);





% Options for prox min
options.stepsize = .45e-4;
options.convTol = 15e-10;
%options.xsize = [256,256];
options.maxIter = 600;
options.residTol = 5e-5;
options.momentum = 'nesterov';
options.disp_figs = 1;
options.disp_fig_interval = 10;   %display image this often
options.print_interval = 10;
options.xsize = size(u_reshaped(:,:,1));
options.known_input = 0;
options.disp_crop = @(x)x;
nocrop = @(x)x;
h1 = figure(1),clf
options.fighandle = h1;
nopad = @(x)x;

tau = 3e-2;
%prox_handle = @(x)deal(max(x,0), 0*tau*sum(abs(vec(x))));
prox_handle = @(x)deal(.5*tv2d_aniso_haar(x,tau*options.stepsize) + ...
        .5*max(x,0), tau*options.stepsize*TVnorm(x));
    


[xhat_svd, f_svd] = proxMin(GradErrHandle,prox_handle,0*u_reshaped(:,:,1),imd,options);



%% Try windowed convolution approach

% Figure out windows
nwiny = 3;
nwinx = 4;
Nwins = nwiny*nwinx;
dy = Ny/(nwiny);
dx = Nx/(nwinx);
ywin = (dy/2:dy:nwiny*dy);
xwin = (dx/2:dx:nwinx*dx);
[Xwin, Ywin] = meshgrid(xwin,ywin);

% Find closest points

[k,d] = dsearchn(fliplr(-si_mat'+[Ny/2,Nx/2]),cat(2,Xwin(:),Ywin(:)));
% k(i) contains indx of psf in yi matrix that is closest to window point
% Xwin(i), Ywin(i)

% Create window functions for each 
yi_winconv = yi_reg(:,:,k);
%%
H_win = zeros(size(nopad(u_reshaped(:,:,1)),1), size(nopad(u_reshaped(:,:,1)),2), rnk);
for r = 1:Nwins
    H_win(:,:,r) = fft2(ifftshift(yi_winconv(:,:,r)));
end
H_win_conj = conj(H_win);


%%


win_stack = zeros(Ny,Nx,Nwins);
win_stack2 = win_stack;
for n = 1:Nwins
    win_vals = zeros(Nwins,1);
    win_vals(n) = 1;
    scaterp = scatteredInterpolant(Xwin(:), Ywin(:), win_vals,'linear','linear'); 
    scaterp1 = scatteredInterpolant(vec(Xwin), vec(Ywin), win_vals,'natural','linear'); 
    win_stack(:,:,n) = scaterp(Xq+Nx/2,Yq+Ny/2);
    win_stack2(:,:,n) = scaterp1(Xq+Nx/2,Yq+Ny/2);
    %win_stack(:,:,n) = griddata(Xwin(:),Ywin(:),win_vals,Xq+Nx/2,Yq+Ny/2,'v4');
    %win_stack(:,:,n) = interp2(Xwin,Ywin,reshape(win_vals,size(Xwin)),Xq+Nx/2,Yq+Ny/2,'linear');
    imagesc(win_stack2(:,:,n))
    title(n)
    axis image
    drawnow
end

%%




%%

A_win = @(x)A_svd(H_win, win_stack, x, nocrop);
A_win_adj = @(x)A_adj_svd(H_win_conj, win_stack, x, nopad);


options_win.stepsize = .45e-4;
options_win.convTol = 15e-10;
%options.xsize = [256,256];
options_win.maxIter = 600;
options_win.residTol = 5e-5;
options_win.momentum = 'nesterov';
options_win.disp_figs = 1;
options_win.disp_fig_interval = 3;   %display image this often
options_win.xsize = size(u_reshaped(:,:,1));
options_win.known_input = 0;
options_win.disp_crop = @(x)x;
nocrop = @(x)x;
h1 = figure(1),clf
options_win.fighandle = h1;
nopad = @(x)x;

GradErrHandle_win = @(x) linear_gradient(x,A_win,A_win_adj,imd);
tau_win = 3e-2;
%prox_handle = @(x)deal(max(x,0), 0*tau*sum(abs(vec(x))));
prox_handle_win = @(x)deal(.5*tv2dApproxHaar(x,tau_win*options.stepsize) + ...
        .5*max(x,0), tau_win*options.stepsize*TVnorm(x));

[xhat_win, f_win] = proxMin(GradErrHandle_win,prox_handle_win,0*u_reshaped(:,:,1),imd,options_win);


