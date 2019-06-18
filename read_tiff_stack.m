function stack = read_tiff_stack(path,varargin)
%stack = read_tiff_stack(path,downsample_ratio)

if nargin>1
    ds = varargin{1};
else
    ds = 1;
end
info = imfinfo(path);
num_images = numel(info);
stack = zeros(info(1).Height/ds, info(1).Width/ds, num_images);
for k = 1:num_images
    stack(:,:,k) = imresize(imread(path, k, 'Info', info),1/ds,'box');
    
end