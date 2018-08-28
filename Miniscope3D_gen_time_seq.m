% Generate locations within volume


Nx = 752/4;
Ny = 480/4;
Nz = 128/2;

FoVx = .6; 
FoVy = Ny/Nx * FoVx;
FoVz = .005*64;

dx = FoVx/Nx;
dy = FoVy/Ny;
dz = FoVz/Nz;
V = (zeros(Ny,Nx,Nz));

N_neurons = 200;
rng(1)
kidx = randsample(Nx*Ny*Nz,N_neurons);
[ky, kx, kz] = ind2sub([Ny, Nx, Nz], kidx);
pos_y= ky*dy-round(Ny/2)*dy;
pos_x = kx*dx-round(Nx/2)*dx;
pos_z = kz*dz-round(Nz/2)*dz;
neural_filt = zeros(5,5,5);
V(kidx) = 1;
show_3d_stack(V)

fid = fopen('~/Documents/neural_points.dat','w');
fprintf(fid,'#x\ty\tz\n');
for n = 1:N_neurons
    fprintf(fid,'%.4e\t%.4e\t%.4e\n',pos_x(n),pos_y(n),pos_z(n));
end
fclose(fid);

%%



% Make time sequences
tau_neural = 5e-3;   %Time constant of 
dt = 1e-3;   % Time grid
T = .1;
t = 0:dt:T;
Nt = length(t);

neural_activity = zeros(N_neurons,Nt);
tidx = randsample(N_neurons*Nt,500);
neural_activity(tidx) = .5*rand(length(tidx),1)+.5;
tfilt = exp(-t/tau_neural);
imagesc(neural_activity)


neural_activity_filt = conv2(neural_activity,tfilt);
neural_activity_filt = neural_activity_filt(:,1:Nt);

imagesc(neural_activity_filt)

%%
 % Loop over neurons to generate PSFs
Nz_stack = size(zstack,3);
pad = @(x)padarray(x,[Ny/2,Nx/2],'both')
crop = @(x)x(Ny/2+1:Ny*3/2,Nx/2+1:Nx*3/2);
psfs = zeros(Ny,Nx,N_neurons);
zstack_resized = max(imresize(zstack_bg,[Ny,Nx],'box')-10,0);
for n = 1:N_neurons
    psfs(:,:,n) = crop(circshift(pad(zstack_resized(:,:,min(kz(n),Nz_stack))),[ky(n)*2-Ny,kx(n)*2-Nx]));
   n
end

%%

% Play back time sequence
n = 0;
h2 = figure(2);
clf
h1 = figure(1);
clf
for t_vid = t
    clf
    n = n+1;
    %V = zeros(Ny,Nx,Nz);
    gry = [.3 .3 .3]/3;
    set(0,'CurrentFigure',h1);
    scatter3(pos_x, pos_y, pos_z, [], neural_activity_filt(:,n))
    cmap_green = [gry(1)*ones(128,1), linspace(gry(2),1,128)', gry(3)*ones(128,1)];
    colormap(cmap_green)
    set(gca,'Color',gry)
    set(gca,'GridColor',[1 1 1])
    set(gcf,'Color',[ 0 0 0])

    svid(:,:,n) = (sum(psfs.*reshape(neural_activity_filt(:,n),[1,1,N_neurons]),3));
    set(0,'CurrentFigure',h2)

    
    
    imagesc(svid(:,:,n))
    drawnow
    %drawnow
    %V(kidx) = neural_activity_filt(:,n);
    %imagesc(sum(V,3));
   % axis image

end

%%

%svid_diff = diff(svid,1,3).*svid(:,:,1:end-1);
svid_diff = svid;
for n = 1:size(svid_diff)
    imagesc(abs(svid_diff(:,:,n)))
    caxis([0 max(svid_diff(:))])
    axis image

    drawnow
    pause(1/10)
end


    



