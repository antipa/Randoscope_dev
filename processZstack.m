hin = load('/Users/nick.antipa/Documents/Diffusers/Miniscope/calibration/h_20180311_for_zebrafish.mat');

%%
h = hin.h;

dsz = 2;
ds = 4;
nzout = floor(size(h,3)/dsz);
hds = zeros(size(h,1)/ds,size(h,2)/ds,nzout);
hdsbg = hds;
hdsnorm = hds;
nx = size(hds,2);
ny = size(hds,1);

divnorm = [];
bgmean = divnorm;
bgmeanmod = divnorm;
h2 = figure(2),clf
h1 = figure(1),clf

[X,Y] = meshgrid(1:size(hds,2),1:size(hds,1));
for n = 1:dsz:dsz*nzout
    stackidx = (n-1)/dsz+1;
    idx =[];
    

    
    hds(:,:,stackidx) = 1/ds*imresize(sum(h(:,:,n:n+dsz),3),1/ds,'box');
    testim = hds(:,:,stackidx);
    thresh = mean2(testim)+2*std2(testim);
    idx = testim>=thresh;
    for m = 1:10
        
        thresh = mean(testim(idx))+2*std(testim(idx));
        idx = testim<=thresh;

        
    end
    set(0,'CurrentFigure',h1)
    
 
    %bg = mean(testim(idx));
    
    %hds(:,:,stackidx) = 1/ds*imresize(medfilt2(sum(h(:,:,n:n+dsz),3),[ds,ds]),1/ds,'box');
    %hds(:,:,stackidx) = 1/ds*imresize(medfilt2(sum(h(:,:,n:n+dsz),3),[ds,ds]),1/ds,'box');
    [y, x] = find(idx);
    y = y-ny/2;
    x = x-nx/2;
    sf = fit([x, y],testim(idx),'poly44');
    bg = sf(X-nx/2,Y-ny/2);
    imz = testim - bg;
    imagesc(imz);
    axis image
    drawnow
    divnorm(stackidx) = norm(imz,'fro');
    hdsnorm(:,:,stackidx) = imz/divnorm(stackidx);
    bgmeanmod(stackidx) = mean2(imz(1:20,1:20));
    bgmean(stackidx) = mean2(testim(1:20,1:20));
    set(0,'CurrentFigure',h2)
    plot(bgmean)
    hold on
    plot(bgmeanmod)
    legend('original','outlier rejection')
    
    xlim([0 nzout])
    hold off
    %imagesc(hds(:,:,stackidx));
    %axis image
    drawnow
    
end

%% iterative outlier rejection
testim = h(:,:,1);
imagesc(testim)
thresh = mean2(testim)+2*std2(testim);
idx = testim>=thresh;
niter = 20;
for n = 1:niter
    thresh = mean(testim(idx))+2*std(testim(idx));
    idx = testim<=thresh;
    imagesc(testim.*(idx));
    axis image
    drawnow
end



%%
imnormalized = imstack;
for m = 1:size(imstack,3)
    m
    imnormalized(:,:,m) = imstack(:,:,m)/divnorm(m);
end