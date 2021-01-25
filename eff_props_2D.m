clear

loadValue = -0.002;
nGrid = 6;
nTimeSteps = 1;
nIter = 1000;

Nx  = 32 * nGrid;     % number of space steps
Ny  = 32 * nGrid;

Sxx = get_sigma_2D(loadValue, [1.0, 1.0, 0], nGrid, nTimeSteps, nIter) / loadValue

% GPU CALCULATION
system(['nvcc -DNGRID=', int2str(nGrid), ' -DNT=', int2str(nTimeSteps), ' -DNITER=', int2str(nIter), ' -DNPARS=', int2str(9), ' boundary_problem.cu']);
system(['.\a.exe']);

fil = fopen('Pm.dat', 'rb');
Pm = fread(fil, 'double');
fclose(fil);
Pm = reshape(Pm, Nx, Ny);

fil = fopen('Pc.dat', 'rb');
Pc = fread(fil, 'double');
fclose(fil);
Pc = reshape(Pc, Nx, Ny);

diffP = Pm - Pc;

fil = fopen('tauXYm.dat', 'rb');
tauXYm = fread(fil, 'double');
fclose(fil);
tauXYm = reshape(tauXYm, Nx - 1, Ny - 1);

fil = fopen('tauXYc.dat', 'rb');
tauXYc = fread(fil, 'double');
fclose(fil);
tauXYc = reshape(tauXYc, Nx - 1, Ny - 1);

diffTauXY = tauXYm - tauXYc;

fil = fopen('tauXYavm.dat', 'rb');
tauXYavm = fread(fil, 'double');
fclose(fil);
tauXYavm = reshape(tauXYavm, Nx, Ny);

fil = fopen('tauXYavc.dat', 'rb');
tauXYavc = fread(fil, 'double');
fclose(fil);
tauXYavc = reshape(tauXYavc, Nx, Ny);

diffTauXYav = tauXYavm - tauXYavc;

fil = fopen('J2m.dat', 'rb');
J2m = fread(fil, 'double');
fclose(fil);
J2m = reshape(J2m, Nx, Ny);

fil = fopen('J2c.dat', 'rb');
J2c = fread(fil, 'double');
fclose(fil);
J2c = reshape(J2c, Nx, Ny);

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

