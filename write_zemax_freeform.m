function write_zemax_freeform(sag,Fx,Fy,Fxy,pixel_size,filename)
[nrows, ncols] = size(sag);
if isempty(Fx) || isempty(Fy) || isempty(Fxy)
    Fx = zeros(size(sag));
    Fy = Fx;
    Fxy = Fx;
end
tstamp = datestr(now,'yyyymmdd_HHMMSS');
fid = fopen([filename,'_',tstamp,'.DAT'],'w');
%USE MILIMETERS
fprintf(fid,'%i\t%i\t%.4e\t%.4e\t%i\t%.4f\t%.4f\n',ncols,nrows,pixel_size,pixel_size,0,0,0);

for m = 1:numel(Fx)
    fprintf(fid,'%.8f\t%.8f\t%.8f\t%.8f\n',sag(m),Fx(m),Fy(m),Fxy(m));
    if mod(m,ncols)==0
        fprintf('%i percent done...\n',round(m/numel(Fx)*100))
    end
end
fclose(fid);