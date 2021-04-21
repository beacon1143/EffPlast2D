clear
figure(1)
clf
colormap jet

loadValue = -0.002;
nGrid = 20;
nTimeSteps = 1;
nIter = 10000000;
eIter = 1.0e-10;
needCPUcalculation = false;

Nx  = 32 * nGrid;     % number of space steps
Ny  = 32 * nGrid;

Sxx = get_sigma_2D(loadValue, [1.0, 1.0, 0], nGrid, nTimeSteps, nIter, eIter, needCPUcalculation);

% GPU CALCULATION
system(['nvcc -DNGRID=', int2str(nGrid), ' -DNT=', int2str(nTimeSteps), ' -DNITER=', int2str(nIter), ' -DEITER=', num2str(eIter), ' -DNPARS=', int2str(10), ' EffPlast2D.cu main.cu']);
system(['.\a.exe']);

fil = fopen('Pc.dat', 'rb');
Pc = fread(fil, 'double');
fclose(fil);
Pc = reshape(Pc, Nx, Ny);

fil = fopen('tauXXc.dat', 'rb');
tauXXc = fread(fil, 'double');
fclose(fil);
tauXXc = reshape(Pc, Nx, Ny);

fil = fopen('tauYYc.dat', 'rb');
tauYYc = fread(fil, 'double');
fclose(fil);
tauYYc = reshape(Pc, Nx, Ny);

fil = fopen('tauXYc.dat', 'rb');
tauXYc = fread(fil, 'double');
fclose(fil);
tauXYc = reshape(tauXYc, Nx - 1, Ny - 1);

fil = fopen('tauXYavc.dat', 'rb');
tauXYavc = fread(fil, 'double');
fclose(fil);
tauXYavc = reshape(tauXYavc, Nx, Ny);

fil = fopen('J2c.dat', 'rb');
J2c = fread(fil, 'double');
fclose(fil);
J2c = reshape(J2c, Nx, Ny);

if needCPUcalculation
  fil = fopen('Pm.dat', 'rb');
  Pm = fread(fil, 'double');
  fclose(fil);
  Pm = reshape(Pm, Nx, Ny);

  diffP = Pm - Pc;

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
  imagesc(tauXYm)
  colorbar
  title('tauXYm')
  axis image

  subplot(2, 2, 4)
  imagesc(diffTauXY)
  colorbar
  title('diffTauXY')
  axis image

  drawnow
else
  % POSTPROCESSING
  subplot(2, 2, 1)
  imagesc(Pc(2:end-1, 2:end-1))
  colorbar
  title('P')
  axis image

  subplot(2, 2, 3)
  imagesc(tauXYc(2:end-1, 2:end-1))
  colorbar
  title('tauXY')
  axis image

  subplot(2, 2, 2)
  imagesc(tauXXc(2:end-1, 2:end-1))
  colorbar
  title('tauXX')
  axis image

  subplot(2, 2, 4)
  imagesc(tauYYc(2:end-1, 2:end-1))
  colorbar
  title('tauYYc')
  axis image

  drawnow
end %if