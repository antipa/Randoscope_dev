function pts = make_3d_pts(num_x, sep_x, num_y, sep_y, num_z, sep_z);
    % A group has a width sep_x * num_x
    % There are N groups length(sep_x)
    % So loop over groups, then within groups loop over elements
    % Creates clusters of points, each being num_x x num_y x num_z points
    % separated by sep_x, sep_y, and sep_z. They'll be arranged in a line,
    % so for N groups, each group will be placed along the x direction
    % evenly distributed with some buffer between groups.
    %
    % returns list of points in whatever units sep_x is in 
    
    N = length(sep_x);   %Get number of groups
    gc = (-N/2+.5:N/2-.5) * 1.1*max(sep_x.*num_x); 
    pts = [];
    for g = 1:length(sep_x)
        nx = num_x(g);
        ny = num_y(g);
        nz = num_z(g);
        sx = sep_x(g);
        sy = sep_y(g);
        sz = sep_z(g);
       
        c = gc(g);
        
        px = (-nx/2+.5 : nx/2-.5)*sx - c;
        py = (-ny/2+.5 : ny/2-.5)*sy;
        pz = (-nz/2+.5 : nz/2 - .5)*sz;
        
        [PX, PY, PZ] = meshgrid(px,py,pz);
        pts = cat(1,pts,cat(2,PX(:),PY(:),PZ(:)));
        
    end
end
