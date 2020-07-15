function success = containers2file(filename,cont)

[FID,success] = fopen(filename,'w');

key_cell = keys(cont);
val_cell = values(cont);

for i = 1:numel(key_cell)
    if isnumeric(val_cell{i})
        val_str = num2str(val_cell{i},'%.6e');
    else
        val_str = val_cell{i};
    end
   fprintf(FID,'%s\t%s\n',key_cell{i},val_str);
end

fclose(FID);
