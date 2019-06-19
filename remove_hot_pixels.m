function not_as_hot = remove_hot_pixels(so_hot_right_now, nhood, thresh)

so_hot_right_now = padarray(so_hot_right_now,[nhood, nhood],'replicate','both');
med = medfilt2(so_hot_right_now,[nhood, nhood]);
hot_pix = abs(so_hot_right_now - med);
bad_pix = (hot_pix >= thresh);
not_as_hot = so_hot_right_now;
not_as_hot(bad_pix) = med(bad_pix);
not_as_hot = not_as_hot(nhood+1:end-nhood,nhood+1:end-nhood);
return
