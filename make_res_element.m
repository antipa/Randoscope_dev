function imout = make_res_element(nbarx, nbarz, pitchx_px, pitchz_px)
    % Create test chart bars with bar width pitchx_px and pitchz_px,
    % repeating nbarx and nbarz times
    
    % Figure out the im size
    FoVx = pitchx_px * (2*nbarx);
    FoVz = pitchz_px * (2*nbarz);
    [X,Z] = meshgrid(0:FoVx-1, 0:FoVz - 1);
    subimx = mod(X,2*pitchx_px)/2/pitchx_px < 0.5;
    subimz = mod(Z,2*pitchz_px)/2/pitchz_px < 0.5;
    imout = cat(2,subimx, subimz);

end