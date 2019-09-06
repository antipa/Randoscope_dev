% Loads in calibration data and removes background

data_dir = 'Q:\kyrollos\RandoscopeNanoscribe\Nanoscribe_pdms\Data_8_21_2019\2_5um_PSF_20um_1'

file_type = '.ome'

files = dir(data_dir);
m = 0;
bck_files = struct('order',[],'name',[],'p',[]);
bck_ind = 0;
psf_ind = 0;
psf_files = struct('name',[],'order',[])
for n = 1:length(files)
    is_ftype = strfind(files(n).name,file_type);
    if ~isempty(is_ftype)
        m = m + 1;
        is_bck = strfind(files(n).name,'BCK');
        is_psf = strfind(files(n).name,'Pos_');
        unders = strfind(files(n).name,'_');
        if ~isempty(is_bck)
            bck_ind = bck_ind+1;
            %fprintf(['BG file: ',files(n).name,'\n'])
            fpost = files(n).name(is_ftype-1);
            p = strcmpi(fpost,'p');
            if p
                bck_num = files(n).name(is_bck+3:is_ftype-2);
            else
                bck_num = files(n).name(is_bck+3:is_ftype-1);
                
            end
            bck_files(bck_ind).order = str2double(bck_num);
            bck_files(bck_ind).name = files(n).name;
            bck_files(bck_ind).p = p;
        elseif ~isempty(is_psf)
            psf_ind = psf_ind+1;
            psf_num = files(n).name(is_psf+4:unders(end-1)-1);
            psf_files(psf_ind).name = files(n).name;
            psf_files(psf_ind).order = str2double(psf_num);
        end
    else
        files(n).name
    end
end
[~,psf_sort] = sort([psf_files(:).order],'ascend');

psf_files_sorted = psf_files(psf_sort);
Nxy = 192;

psfs = struct('xystack',[]);
ds = 2;
info = imfinfo([data_dir,'\',psf_files(1).name]);
Ny = info(1).Height/ds;
Nx = info(1).Width/ds;
Nc = info(1).SamplesPerPixel;
%

[~,bck_sort] = sort([bck_files(:).order],'ascend');
bck_files_sorted = bck_files(bck_sort);

for n = 1:length(bck_files)
    n
    bck_files_sorted(n).bck = mean(read_tiff_stack([data_dir,'\',bck_files_sorted(n).name],2),4);
end


rnk=24;
comps_out = zeros(Ny,Nx,rnk,1);
weights_out = zeros(Ny,Nx,rnk,1);
%% Prepare list of files per depth
zcount = 0;
psf_files_z = struct('XYnames',{});

for n = 1:length(psf_files_sorted)
    
    if mod(n,Nxy)==1
        zcount = zcount+1;
        xycount = 0;
        
    end
    
    
    
    
    %     if psf_files_sorted(n).order ~= 6721
    % Load and average in time dimension
    xycount=xycount + 1;
    psf_files_z(zcount).XYnames{xycount} = psf_files_sorted(n).name;
    %     end
    %psf_in = squeeze(mean(read_tiff_stack([data_dir,'\',psf_files_sorted(n).name],ds),4));
    
    %
    
    %     if psf_files_sorted(n).order == 6721
    %         xycount = xycount+1;
    %     end
    
end



%%

for zcount = 1:length(psf_files_z)
    psfs.xystack = zeros(Ny,Nx,Nc,Nxy);
    for xycount = 1:length(psf_files_z(zcount).XYnames)
        psf_in = squeeze(mean(read_tiff_stack([data_dir,'\',psf_files_z(zcount).XYnames{xycount}],ds),4));
        psfs.xystack(:,:,:,xycount) = psf_in;
        fprintf('Z: %i \t XY %i\n',zcount,xycount)
        fprintf(['image ',psf_files_z(zcount).XYnames{xycount},'\n'])
        imagesc(psfs.xystack(:,:,:,xycount)/100), axis image
        title(sprintf('Z %i, xyindex %i',zcount,xycount))
        drawnow
    end
    icenter = 89;   %Location of reference image in each xy layer
    colors = 2;   %Use color index (1 for r, 2 for g, 3 for b). String 'all' will average.
    if strcmpi(colors,'all')
        stack_mono = squeeze(mean(double(psfs.xystack) - bck_files_sorted(2*zcount).bck,3));   %2 because there are 2 bck files per z plane
        ref_im = mean(psfs.xystack(:,:,:,icenter) - bck_files_sorted(2*zcount-1).bck,3);
    else
       
        stack_mono = squeeze(double(psfs.xystack(:,:,colors,:)) - bck_files_sorted(2*zcount).bck(:,:,colors));
        ref_im = psfs.xystack(:,:,colors,icenter) - bck_files_sorted(2*zcount-1).bck(:,:,colors);
    end

   
    
    [comps, weights_interp, weights,shifts,yi_reg_out] = Miniscope_svd_xy(stack_mono,ref_im,rnk,'boundary_condition','circular');
    if zcount == 1
        comps_out(:,:,:,1) = comps;
        weights_out(:,:,:,1) = weights_interp;
        axial_stack = ref_im;
    else
        comps_out = cat(4,comps_out,comps);
        weights_out = cat(4,weights_out,weights_interp);
        axial_stack = cat(3,axial_stack,ref_im);
    end
end

% for n = 1:length(psf_files_sorted)
%
%     if mod(n,Nxy)==1
%         zcount = zcount+1;
%         psfs.xystack = zeros(Ny,Nx,Nc,Nxy);
%         xycount = 0;
%
%     end
%
%
%
%     xycount=xycount + 1;
%     % Load and average in time dimension
%     %psf_file_z(zcount).
%     %psf_in = squeeze(mean(read_tiff_stack([data_dir,'\',psf_files_sorted(n).name],ds),4));
%
%     %
%     psfs.xystack(:,:,:,xycount) = psf_in;
%     imagesc(psfs.xystack(:,:,:,xycount)/100), axis image
%     title(sprintf('Image number %i, xyindex %i',n,xycount))
%     drawnow
%     if psf_files_sorted(n).order == 6336
%         xycount = xycount+1;
%     end
%     if xycount == Nxy
%         % Average across color channels to make 3d stack
%         stack_mono = squeeze(mean(double(psfs.xystack) - bck_files_sorted(2*zcount).bck,3));   %2 because there are 2 bck files per z plane
%         icenter = 89;
%         ref_im = mean(double(psfs.xystack(:,:,:,icenter) - bck_files_sorted(2*zcount-1).bck),3);
%
%         [comps, weights_interp, weights,shifts,yi_reg_out] = Miniscope_svd_xy(stack_mono,ref_im,rnk,'boundary_condition','circular');
%         if zcount == 1
%             comps_out(:,:,:,1) = comps;
%             weights_out(:,:,:,1) = weights_interp;
%             axial_stack = ref_im;
%         else
%             comps_out = cat(4,comps_out,comps);
%             weights_out = cat(4,weights_out,weights_interp);
%             axial_stack = cat(3,axial_stack,ref_im);
%         end
%
%     end
% end
%%
imagesc(axial_stack(:,:,1))

axis image
drawnow

% for n = 1:length(bck_files)
%     n
%     bck_files_sorted(n).bck = mean(double(read_tiff_stack([data_dir,'\',bck_files_sorted(n).name],2)),4);
% end

%%
% for n = 1:length(psf_files_sorted)
%    psf_files_sorted(n).name
%    pause(1/30)
% end
