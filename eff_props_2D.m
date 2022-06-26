clear
figure(1)
clf
colormap jet

loadValue = -0.0004;
loadType = [1.0, 1.0, 0.0];
nGrid = 24;
nTimeSteps = 2;
nIter = 100000;
eIter = 1.0e-10;
needCPUcalculation = false;
needCompareStatic = true;

Nx  = 32 * nGrid;     % number of space steps
Ny  = 32 * nGrid;

Sxx = get_sigma_2D(loadValue, loadType, nGrid, nTimeSteps, nIter, eIter, needCPUcalculation);

% GPU CALCULATION
system(['rm a.*']);
system(['nvcc -DNGRID=', int2str(nGrid), ' -DNT=', int2str(nTimeSteps), ' -DNITER=', int2str(nIter), ' -DEITER=', num2str(eIter), ' -DNPARS=', int2str(11), ' EffPlast2D.cu main.cu']);
system(['.\a.exe ', num2str(loadValue), ' ', num2str(loadType(1)), ' ', num2str(loadType(2)), ' ', num2str(loadType(3))]);

fil = fopen(strcat('Pc_', int2str(Nx), '_.dat'), 'rb');
Pc = fread(fil, 'double');
fclose(fil);
Pc = reshape(Pc, Nx, Ny);
Pc = transpose(Pc);

fil = fopen(strcat('tauXXc_', int2str(Nx), '_.dat'), 'rb');
tauXXc = fread(fil, 'double');
fclose(fil);
tauXXc = reshape(tauXXc, Nx, Ny);
tauXXc = transpose(tauXXc);

fil = fopen(strcat('tauYYc_', int2str(Nx), '_.dat'), 'rb');
tauYYc = fread(fil, 'double');
fclose(fil);
tauYYc = reshape(tauYYc, Nx, Ny);
tauYYc = transpose(tauYYc);

fil = fopen(strcat('tauXYc_', int2str(Nx), '_.dat'), 'rb');
tauXYc = fread(fil, 'double');
fclose(fil);
tauXYc = reshape(tauXYc, Nx - 1, Ny - 1);
tauXYc = transpose(tauXYc);

fil = fopen(strcat('tauXYavc_', int2str(Nx), '_.dat'), 'rb');
tauXYavc = fread(fil, 'double');
fclose(fil);
tauXYavc = reshape(tauXYavc, Nx, Ny);
tauXYavc = transpose(tauXYavc);

fil = fopen(strcat('J2c_', int2str(Nx), '_.dat'), 'rb');
J2c = fread(fil, 'double');
fclose(fil);
J2c = reshape(J2c, Nx, Ny);
J2c = transpose(J2c);

fil = fopen(strcat('Uxc_', int2str(Nx), '_.dat'), 'rb');
Uxc = fread(fil, 'double');
fclose(fil);
Uxc = reshape(Uxc, Nx + 1, Ny);
Uxc = transpose(Uxc);

fil = fopen(strcat('Uyc_', int2str(Nx), '_.dat'), 'rb');
Uyc = fread(fil, 'double');
fclose(fil);
Uyc = reshape(Uyc, Nx, Ny + 1);
Uyc = transpose(Uyc);

%Ur = sqrt(Ux(1:end-1,:) .* Ux(1:end-1,:) + Uy(:,1:end-1) .* Uy(:,1:end-1))

if needCPUcalculation
  fil = fopen('Pm.dat', 'rb');
  Pm = fread(fil, 'double');
  fclose(fil);
  Pm = reshape(Pm, Nx, Ny);

  diffP = Pm - Pc;
  
  fil = fopen('tauXXm.dat', 'rb');
  tauXXm = fread(fil, 'double');
  fclose(fil);
  tauXXm = reshape(tauXXm, Nx, Ny);

  diffTauXX = tauXXm - tauXXc;
  
  fil = fopen('tauYYm.dat', 'rb');
  tauYYm = fread(fil, 'double');
  fclose(fil);
  tauYYm = reshape(tauYYm, Nx, Ny);

  diffTauYY = tauYYm - tauYYc;

  fil = fopen('tauXYm.dat', 'rb');
  tauXYm = fread(fil, 'double');
  fclose(fil);
  tauXYm = reshape(tauXYm, Nx - 1, Ny - 1);

  diffTauXY = tauXYm - tauXYc;

  fil = fopen('tauXYavm.dat', 'rb');
  tauXYavm = fread(fil, 'double');
  fclose(fil);
  tauXYavm = reshape(tauXYavm, Nx, Ny);

  diffTauXYav = tauXYavm - tauXYavc;

  fil = fopen('J2m.dat', 'rb');
  J2m = fread(fil, 'double');
  fclose(fil);
  J2m = reshape(J2m, Nx, Ny);

  diffJ2 = J2c - J2m;

  % POSTPROCESSING
  subplot(2, 2, 1)
  imagesc(Pm)
  colorbar
  title('Pm')
  axis image

  subplot(2, 2, 3)
  imagesc(diffP)
  colorbar
  title('diffP')
  axis image

  subplot(2, 2, 2)
  imagesc(tauXXm)
  colorbar
  title('tauXXm')
  axis image

  subplot(2, 2, 4)
  imagesc(diffTauYY)
  colorbar
  title('diffTauYY')
  axis image

  drawnow
else
  % POSTPROCESSING
  if needCompareStatic
    % ANALYTIC SOLUTION FOR STATICS

    fil = fopen(strcat('Uanr_', int2str(Nx), '_.dat'), 'rb');
    Uanr = fread(fil, 'double');
    fclose(fil);
    Uanr = reshape(Uanr, Nx, Ny);
    
    fil = fopen(strcat('Unur_', int2str(Nx), '_.dat'), 'rb');
    Unur = fread(fil, 'double');
    fclose(fil);
    Unur = reshape(Unur, Nx, Ny);

    fil = fopen(strcat('Sanrr_', int2str(Nx), '_.dat'), 'rb');
    Sanrr = fread(fil, 'double');
    fclose(fil);
    Sanrr = reshape(Sanrr, Nx - 1, Ny - 1);
    
    fil = fopen(strcat('Sanff_', int2str(Nx), '_.dat'), 'rb');
    Sanff = fread(fil, 'double');
    fclose(fil);
    Sanff = reshape(Sanff, Nx - 1, Ny - 1);

    fil = fopen(strcat('Snurr_', int2str(Nx), '_.dat'), 'rb');
    Snurr = fread(fil, 'double');
    fclose(fil);
    Snurr = reshape(Snurr, Nx - 1, Ny - 1);
    
    fil = fopen(strcat('Snuff_', int2str(Nx), '_.dat'), 'rb');
    Snuff = fread(fil, 'double');
    fclose(fil);
    Snuff = reshape(Snuff, Nx - 1, Ny - 1);
    
    fil = fopen(strcat('plast_', int2str(Nx), '_.dat'), 'rb');
    plast = fread(fil, 'double');
    fclose(fil);
    plast = reshape(plast, Nx - 1, Ny - 1);

    % POSTPROCESSING
    subplot(3, 4, 1)
    imagesc(Snurr)
    colorbar
    title('\sigma_{rr} numerical')
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    
    subplot(3, 4, 2)
    imagesc(Snuff)
    colorbar
    title('\sigma_{\phi \phi} numerical')
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    
    subplot(3, 4, 3)
    imagesc(Unur)
    colorbar
    title('U_{r} numerical')
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    
    subplot(3, 4, 5)
    imagesc(Sanrr)
    colorbar
    title('\sigma_{rr} analytics')
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    
    subplot(3, 4, 6)
    imagesc(Sanff)
    colorbar
    title('\sigma_{\phi \phi} analytics')
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')

    eps = 10e-18;
    
    if abs(loadType(1) - loadType(2)) < eps && abs(loadType(3)) < eps
      subplot(3, 4, 7)
      imagesc(Uanr)
      colorbar
      title('U_{r} analytics')
      axis image
      set(gca, 'FontSize', 10, 'fontWeight', 'bold')

      errorUr = zeros(Nx, Ny);
    end %if
    
    errorSrr =  zeros(Nx - 1, Ny - 1);
    errorSff =  zeros(Nx - 1, Ny - 1);
    
    for i = 1: Nx
      for j = 1: Ny

        if i < Nx && j < Ny

          if abs(Sanrr(j, i)) > eps
            errorSrr(j, i) = abs((Snurr(j, i) - Sanrr(j, i)) / Sanrr(j, i));
          end %if
          
          if abs(Sanff(j, i)) > eps
            errorSff(j, i) = abs((Snuff(j, i) - Sanff(j, i)) / Sanff(j, i));
          end %if

        end %if
        
        if abs(loadType(1) - loadType(2)) < eps && abs(loadType(3)) < eps
          if abs(Uanr(j, i)) > eps
            errorUr(j, i) = abs((Unur(j, i) - Uanr(j, i)) / Uanr(j, i));
          end %if
        end %if
        
      end %for
    end %for
    
    maxErrorSrr = max(errorSrr(:))
    maxErrorSff = max(errorSff(:))

    if abs(loadType(1) - loadType(2)) < eps && abs(loadType(3)) < eps
      maxErrorUr = max(errorUr(:))
    end %if
    
    subplot(3, 4, 9)
    imagesc(errorSrr)
    colorbar
    title('\sigma_{rr} error')
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    
    subplot(3, 4, 10)
    imagesc(errorSff)
    colorbar
    title('\sigma_{\phi \phi} error')
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    
    if abs(loadType(1) - loadType(2)) < eps && abs(loadType(3)) < eps
      subplot(3, 4, 11)
      imagesc(errorUr)
      colorbar
      title('U_{r} error')
      axis image
      set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    end %if
    
    subplot(3, 4, 12)
    imagesc(plast)
    colorbar
    title('plast zone')
    axis image
    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
    
    drawnow
  else  
    subplot(2, 3, 1)
    imagesc(Pc(2:end-1, 2:end-1))
    colorbar
    title('P')
    axis image

    subplot(2, 3, 5)
    imagesc(tauXXc(2:end-1, 2:end-1))
    colorbar
    title('tauXX')
    axis image

    subplot(2, 3, 2)
    imagesc(J2c(2:end-1, 2:end-1))
    colorbar
    title('J2')
    axis image

    subplot(2, 3, 4)
    imagesc(tauYYc(2:end-1, 2:end-1))
    colorbar
    title('tauYY')
    axis image
    
    subplot(2, 3, 3)
    imagesc(Uxc)
    colorbar
    title('Ux')
    axis image
    
    subplot(2, 3, 6)
    imagesc(Uyc)
    colorbar
    title('Uy')
    axis image

    drawnow
  end %if (needCompareStatic)
end %if (needCPUcalculation)