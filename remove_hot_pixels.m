function not_as_hot = remove_hot_pixels(so_hot_right_now, nhood, thresh)

med = medfilt2(so_hot_right_now,[nhood, nhood]);
hot_pix = abs(so_hot_right_now - med);
bad_pix = (hot_pix >= thresh);
not_as_hot = so_hot_right_now;
not_as_hot(bad_pix) = med(bad_pix);
return
