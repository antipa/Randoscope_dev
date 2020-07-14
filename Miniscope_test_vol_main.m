xlocs = [2.7 3.5 7];
zlocs = xlocs*5.55;
pts = make_3d_pts([7 7 7],xlocs , [1 1 1], [5 4 3], 2*200./zlocs ,zlocs);
stackout = Miniscope_create_test_vol(px_obj*512,px_obj*512, 360, 200, px_obj, 5, 2*px_obj, 10,pts);