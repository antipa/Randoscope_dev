% use make_res_element.m to create usaf-like bar elements, then resample
% onto grid relevant to out system.
px = 4.541;  %Pixel size in microns/pixel in sensor space
Mag = 5.2;   %Magnification
dx = px/Mag;   %Object space microns/pixel
dz = 5;   %microns

FoVx = 512*dx;
FoVy = 512*dx;
FoVz = 72*dz;

Ny = round(FoVy / dx);
Nx = round(FoVx / dx);
Nz = round(FoVz / dz);

zcent = 200;

grid_x = [-Nx/2+.5 : Nx/2-.5]*dx;  %microns
grid_y = [-Ny/2+.5 : Ny/2-.5]*dx;
grid_z = [0:Nz-1]*dz - zcent;

[X,Y,Z] = meshgrid(grid_x,grid_y,grid_z);

upsamp = 5; %upsample rate for creating bars. 
dxr = dx/upsamp;
dzr = dz/upsamp;

% Make these monotically increasing!
barwidth_um = [2.78 3.5 7];
bardepth_um = [7.5 10 15];

barwidth = barwidth_um/dxr;
bardepth = bardepth_um/dzr;

px_precompute = cell(1,numel(barwidth_um));

for n = 1:numel(barwidth_um)
    px_precompute{n} = double(make_res_element(3,3,barwidth(n),bardepth(n)));
end

xz = zeros(Nz,Nx);

ngz = 4;

zextent_r = size(px_precompute{end},1);  % in pixels
zextent_um = zextent*dzr;
xextent_r = size(px_precompute{end},2);
xextent_um = xextent_r*dxr;

zstart_um = [0 105-zextent_um/2 200-zextent_um/2 360-zextent_um*4.2];
xstart_um = [-1.5*xextent_um -.5*xextent_um .5*xextent_um];

zstart_r = max(round(zstart_um/dzr),1);
xstart_r = max(round(xstart_um/dxr)+upsamp*Nx/2,1);

% Create high res version of image
Nxr = Nx*upsamp;
Nzr = Nz*upsamp;
slice_r = zeros(Nzr,Nxr,numel(xstart_r));
group_start_um = {};
for zi = 1:length(zstart_r)
    for xi = 1:length(xstart_r)
        Dz = size(px_precompute{xi},1);
        Dx = size(px_precompute{xi},2);
        z = zstart_r(zi)+zextent_r/2;
        x = xstart_r(xi)+xextent_r/2;
        group_start_um{zi,xi} = [z*dzr,x*dxr];
        slice_r(z-ceil(Dz/2):z+floor(Dz/2)-1,x-ceil(Dx/2):x+floor(Dx/2)-1,xi) = px_precompute{xi};        
        imagesc(slice_r)
        axis image
        drawnow
    end
end
slice_r = slice_r(1:Nzr,1:Nxr,:);
%%
pd = 4;
Ny_slice = max(ceil(barwidth))+pd;

slab = zeros(Ny_slice,Nxr,Nzr);

for n = (pd/2+1):size(slab,1)-pd/2
    sl = slice_r(:,:,3);
    slab(n,:,:) = slab(n,:,:) + ...
        reshape(sl',[1,size(slice_r,2),size(slice_r,1)]);
end
pd2 = pd + ceil(max(barwidth) - barwidth(2));
for n = floor((pd2/2+1):size(slab,1)-pd2/2)
    sl = slice_r(:,:,2);
    slab(n,:,:) = slab(n,:,:) + ...
        reshape(sl',[1,size(slice_r,2),size(slice_r,1)]);
end

pd1 = pd + ceil(max(barwidth) - barwidth(1));
for n = floor((pd1/2+1):size(slab,1)-pd1/2)
    sl = slice_r(:,:,1);
    slab(n,:,:) = slab(n,:,:) + ...
        reshape(sl',[1,size(slice_r,2),size(slice_r,1)]);
end

slice_out = imresize(slice_r,1/upsamp,'box');
slice_out = slice_out(1:Nz,1:Nx,:);



slab_out = imresize3_box(slab,round(size(slab)/upsamp));



%%
stackout = zeros(Ny,Nx,Nz);
stackout(1+Ny/2-ceil(size(slab_out,1)/2):Ny/2+floor(size(slab_out,1)/2),:,:) = slab_out;

imagesc(squeeze(max(stackout,[],3)));
axis image

% Then resample to target grid
% 
% figure(1),clf
% imagesc(imresize(px_precompute{1},1/upsamp,'box'))
% axis image

    
    
    
    
    