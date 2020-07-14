function vol_out = Miniscope_create_test_vol(FoVx,FoVy,FoVz,zcent,dx,dz,fwhm_x,fwhm_z,pts)
% inputs for FoV and centering are all in microns. 
% dx is the lateral pixel size
% dz is axial pixel size
% fwhm_x and z are the fwhm of the blobs to place (in microns)
% pts is location of points. Use make_3d_pts.m, for example.



% M = 5.2;
% dx = 4/M;  %Microns
% dz = 5;   %Microns

Ny = round(FoVy / dx);
Nx = round(FoVx / dx);
Nz = round(FoVz / dz);

% fwhm_x = 2;   %microns
% fwhm_z = 20;  %microns

sig_x = fwhm_x/2.355;
sig_z = fwhm_z/2.355;

grid_x = [-Nx/2+.5 : Nx/2-.5]*dx;  %microns
grid_y = [-Ny/2+.5 : Ny/2-.5]*dx;
grid_z = [0:Nz-1]*dz - zcent;

[X,Y,Z] = meshgrid(grid_x,grid_y,grid_z);

blobkernel = @(x,y,z)exp(- (X-x).^2/(2*sig_x^2) - (Y-y).^2/(2*sig_x^2) - (Z-z).^2/(2*sig_z^2));

vol_out = make_vol(blobkernel,pts,Nx,Ny,Nz);
end

function vol = make_vol(blobhandle,coords_3d,Nx,Ny,Nz)
    % Pass in Nx3 array of points to make in 3D
    vol = zeros(Ny,Nx,Nz);
    for n = 1:size(coords_3d,1)
        vol = vol + blobhandle(coords_3d(n,1), coords_3d(n,2), coords_3d(n,3));
        imagesc(squeeze(max(vol,[],1)));
        
        axis image
        drawnow
    end
end