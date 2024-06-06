function [field] = read_data_2D(file_name, size, nX, nY)
  fil = fopen(strcat(file_name, '_', int2str(size), '_.dat'), 'rb');
  field = fread(fil, 'double');
  fclose(fil);
  field = reshape(field, nX, nY);
  field = transpose(field);
end % function