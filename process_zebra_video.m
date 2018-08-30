vtest = VideoReader('/Users/nick.antipa/Documents/Diffusers/Miniscope/wormdata/worm_gaussian_2018-07-18.avi');
viddata = vtest.read;
viddata = viddata(:,:,:,1:end-4);


%%
vid_array = double(squeeze(viddata));

tds = 5;
m = 1;

vid_ds = zeros(size(vid_array,1),size(vid_array,2),size(vid_array,3)/tds);
for n = 1:size(vid_array,3)-1
    if mod(n,tds)==0
        m = m+1
    end
    vid_ds(:,:,m) =  vid_ds(:,:,m) + vid_array(:,:,n)/tds;
end


%%
ds = 2;
vid_ads = zeros(size(vid_ds,1)/ds,size(vid_ds,2)/ds,size(vid_ds,3));
for n= 1:size(vid_ads,3)
    vid_ads(:,:,n) = imresize(vid_ds(:,:,n),1/ds,'box');
    imagesc(vid_ads(:,:,n));
    
    axis image
    drawnow
    colormap gray
end
h1 = figure(1); clf
h2 = figure(2); clf

%%
vec = @(x)reshape(x,[numel(x),1]);
immat = zeros(size(vid_ads,1)*size(vid_ads,2),size(vid_ads,3));
for n = 1:size(vid_ads,3)
    immat(:,n) = vec(vid_ads(:,:,n));
end
    
rk = 1;    %Rank of bg
[u,s,v] = svds(immat,rk);
bgim = zeros(size(vid_ads,1),size(vid_ads,2));
for n = 1:rk
    bgim = bgim + reshape(s(n,n)*u(:,n),[size(vid_ads,1),size(vid_ads,2)]);
%     imagesc(reshape(s(n,n)*u(:,n),[size(vid_ads,1),size(vid_ads,2)]));
%     axis image
%     drawnow
%     pause(1)
    
end

vidbg = 0*vid_ads; 

for n = 1:size(vid_ads,3)
    set(0,'CurrentFigure',h1);
    vidbg(:,:,n) = vid_ads(:,:,n) - v(n)*bgim;
    imagesc(abs(vidbg(:,:,n)))
    title('bg removed')
    axis image
    colormap parula

    drawnow
    
    set(0,'CurrentFigure',h2)
    imagesc(vid_ads(:,:,n))
    axis image
    title('original')
    drawnow
    
end



%%
bx = 100;
by = 100;
for n = 1:size(vid_ads,3)
    %imagesc(vidbg(end-50:end,end-50:end,n))
    patch = vidbg(end-by:end,end-bx:end,n);
    if n == 1
        noise_vol = patch;
    else
        noise_vol = cat(3,noise_vol,patch);
    end
    
    %imagesc(real(fftshift(ifft2(fft2(patch).*conj(fft2(patch))))));
    %axis image
    %drawnow
    %pause(1/30)
end

varim = std(vidbg,0,3).^2-std(noise_vol(:)).^2;
stdim = std(vidbg,0,3) - std(noise_vol(:));
imagesc(varim), colorbar
axis image


