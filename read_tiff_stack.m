function stack = read_tiff_stack(path,varargin)
%stack = read_tiff_stack(path,downsample_ratio,list_of_images)
%list_of_images is a vector of indices to read out. if empty, read all.



info = imfinfo(path);
num_images = numel(info);

if nargin>1
    ds = varargin{1};
    if nargin > 2
        k_list = varargin{2};
    else
        k_list = 1:num_images;
    end
else
    ds = 1;
end

stack = zeros(info(1).Height/ds, info(1).Width/ds, num_images);
for k = k_list
    stack(:,:,k) = imresize(imread(path, k, 'Info', info),1/ds,'box');
    
end