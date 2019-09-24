folder = uigetdir('.','Pick folder containing mat files');
%%
out_folder = uigetdir(folder,'Pick output folder');

files = dir(folder);
good_count = 0;
t_ind = []
for n = 1:length(files)
    dots = strfind(files(n).name,'.');
    fext = files(n).name(dots(end):end);
    if strcmpi(fext,'.mat')
        good_file(n) = 1;
        good_count = good_count + 1;
        unders = strfind(files(n).name,'_');
        t_ind(good_count) = str2double(files(n).name(unders(end)+1:dots(end)-1));
    end
end

files_good = files(logical(good_file));



[~,sort_ids] = sort(t_ind,'ascend');

files_sorted = files_good(sort_ids);
Nt = length(files_sorted);


for n = 1:Nt
    file_in = load([files_sorted(n).folder,'/',files_sorted(n).name],'xhat_out');
    n
    if n == 1
        [Ny,Nx,Nz] = size(file_in.xhat_out);
        full_array = single(zeros(Ny,Nx,Nz,Nt));
    end
    full_array(:,:,:,n) = file_in.xhat_out;
end


full_norm = max(max(max(max(full_array))));


array_normed = min(1.1*full_array/full_norm,1);


for t = 1:Nt
    for z = 1:Nz
        imwrite(squeeze(array_normed(:,:,z,t)),[out_folder,sprintf('/waterbear_closer_3_T_%03d_Z_%03d.png',t,z)]);
        z
        t
    end
end
    

    