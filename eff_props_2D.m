clear
figure(1)
clf
colormap jet

initLoadValue = -0.000015;
addLoadValueStep = -0.000025;
loadType = [4.0, -2.0, 0.0];
nGrid = 16;
nTimeSteps = 1;
nTasks = 2;
nIter = 500000;
eIter = 1.0e-10;
N = 2;
needCPUcalculation = false;
needCompareStatic = true;
if N > 1
  needCompareStatic = false;
end %if

Nx  = 32 * nGrid;     % number of space steps
Ny  = 32 * nGrid;

Sxx = get_sigma_2D(addLoadValueStep, loadType, nGrid, nTimeSteps, nIter, eIter, N, needCPUcalculation);

% GPU CALCULATION
system(['nvcc -O 3 -DNL=', int2str(nTasks), ' -DNGRID=', int2str(nGrid), ' -DNITER=', int2str(nIter), ' -DEITER=', num2str(eIter), ' -DNPARS=', int2str(11), ' EffPlast2D.cu main.cu']);
system(['.\a.exe ', num2str(initLoadValue), ' ', num2str(loadType(1)), ' ', num2str(loadType(2)), ' ', num2str(loadType(3)), ' ', num2str(nTimeSteps), ' ' num2str(addLoadValueStep)]);

cd data

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
    fil = fopen(strcat('UnuAbs_', int2str(Nx), '_.dat'), 'rb');
    UnuAbs = fread(fil, 'double');
    fclose(fil);
    UnuAbs = reshape(UnuAbs, Nx, Ny);

    fil = fopen(strcat('J1nu_', int2str(Nx), '_.dat'), 'rb');
    J1nu = fread(fil, 'double');
    fclose(fil);
    J1nu = reshape(J1nu, Nx - 1, Ny - 1);
    
    fil = fopen(strcat('J2nu_', int2str(Nx), '_.dat'), 'rb');
    J2nu = fread(fil, 'double');
    fclose(fil);
    J2nu = reshape(J2nu, Nx - 1, Ny - 1);

    fil = fopen(strcat('plast_nu_', int2str(Nx), '_.dat'), 'rb');
    plast_nu = fread(fil, 'double');
    fclose(fil);
    plast_nu = reshape(plast_nu, Nx - 1, Ny - 1);
    
    subplot(3, 4, 1)
    imagesc(J1nu)
    colorbar
    title('J1 numerical')
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    
    subplot(3, 4, 2)
    imagesc(J2nu)
    colorbar
    title('J2 numerical')
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    
    subplot(3, 4, 3)
    imagesc(plast_nu)
    colorbar
    title('plast zone numerical')
    axis image
    set(gca, 'FontSize', 10)

    subplot(3, 4, 4)
    imagesc(UnuAbs)
    colorbar
    title('abs(U) numerical')
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    
    if N==1
      fil = fopen(strcat('UanAbs_', int2str(Nx), '_.dat'), 'rb');
      UanAbs = fread(fil, 'double');
      fclose(fil);
      UanAbs = reshape(UanAbs, Nx, Ny);
      
      fil = fopen(strcat('errorUabs_', int2str(Nx), '_.dat'), 'rb');
      errorUabs = fread(fil, 'double');
      fclose(fil);
      errorUabs = reshape(errorUabs, Nx, Ny);
      
      fil = fopen(strcat('J1an_', int2str(Nx), '_.dat'), 'rb');
      J1an = fread(fil, 'double');
      fclose(fil);
      J1an = reshape(J1an, Nx - 1, Ny - 1);
      
      fil = fopen(strcat('J2an_', int2str(Nx), '_.dat'), 'rb');
      J2an = fread(fil, 'double');
      fclose(fil);
      J2an = reshape(J2an, Nx - 1, Ny - 1);
      
      fil = fopen(strcat('errorJ1_', int2str(Nx), '_.dat'), 'rb');
      errorJ1 = fread(fil, 'double');
      fclose(fil);
      errorJ1 = reshape(errorJ1, Nx - 1, Ny - 1);
      
      fil = fopen(strcat('errorJ2_', int2str(Nx), '_.dat'), 'rb');
      errorJ2 = fread(fil, 'double');
      fclose(fil);
      errorJ2 = reshape(errorJ2, Nx - 1, Ny - 1);
      
      fil = fopen(strcat('plast_an_', int2str(Nx), '_.dat'), 'rb');
      plast_an = fread(fil, 'double');
      fclose(fil);
      plast_an = reshape(plast_an, Nx - 1, Ny - 1);
      
      plastDiff =  abs(plast_an - plast_nu);

      subplot(3, 4, 5)
      imagesc(J1an)
      colorbar
      title('J1 analytics')
      axis image
      set(gca, 'FontSize', 10, 'fontWeight', 'bold')
      
      subplot(3, 4, 6)
      imagesc(J2an)
      colorbar
      title('J2 analytics')
      axis image
      set(gca, 'FontSize', 10, 'fontWeight', 'bold')
  
      subplot(3, 4, 7)
      imagesc(plast_an)
      colorbar
      title('plast zone analytics')
      axis image
      set(gca, 'FontSize', 10)
      
      subplot(3, 4, 8)
      imagesc(UanAbs)
      colorbar
      title('abs(U) analytics')
      axis image
      set(gca, 'FontSize', 10, 'fontWeight', 'bold')
  
      subplot(3, 4, 9)
      imagesc(errorJ1)
      colorbar
      title('J1 error')
      axis image
      set(gca, 'FontSize', 10, 'fontWeight', 'bold')
      
      subplot(3, 4, 10)
      imagesc(errorJ2)
      colorbar
      title('J2 error')
      axis image
      set(gca, 'FontSize', 10, 'fontWeight', 'bold')
  
      subplot(3, 4, 11)
      imagesc(plastDiff)
      colorbar
      title('plast zone diff')
      axis image
      set(gca, 'FontSize', 10)
      
      subplot(3, 4, 12)
      imagesc(errorUabs)
      colorbar
      title('abs(U) error')
      axis image
      set(gca, 'FontSize', 10, 'fontWeight', 'bold')
    end
    
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

cd ..