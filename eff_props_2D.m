clear

loadValue = 0.00075;
nGrid = 6;
nTimeSteps = 1;
nIter = 1000;

Nx  = 32 * nGrid;     % number of space steps
Ny  = 32 * nGrid;

%Sxx = get_sigma_2D(loadValue, [1, 0, 0], nGrid, nTimeSteps, nIter);
%Syy = get_sigma_2D(loadValue, [0, 1, 0], nGrid, nTimeSteps, nIter);
%Sxy = get_sigma_2D(loadValue, [0, 0, 1], nGrid, nTimeSteps, nIter);

%C1111 = zeros(nTimeSteps, 1);
%C1122 = zeros(nTimeSteps, 1);
%C1112 = zeros(nTimeSteps, 1);
%C2222 = zeros(nTimeSteps, 1);
%C1222 = zeros(nTimeSteps, 1);
%C1212 = zeros(nTimeSteps, 1);

%for it = 1:nTimeSteps
%  C1111(it) = Sxx(it, 1) / loadValue / it * nTimeSteps
%  C1122(it) = Sxx(it, 2) / loadValue / it * nTimeSteps
%  C1112(it) = Sxx(it, 3) / loadValue / it * nTimeSteps
%
%  C2222(it) = Syy(it, 2) / loadValue / it * nTimeSteps
%  C1222(it) = Syy(it, 3) / loadValue / it * nTimeSteps

%  C1212(it) = Sxy(it, 3) / loadValue / it * nTimeSteps
%endfor

Sxx = get_sigma_2D(loadValue, [1, 1, 0], nGrid, nTimeSteps, nIter) / loadValue

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

% POSTPROCESSING
subplot(2, 2, 1)
imagesc(Pm)
colorbar
title('Pm')
axis image

subplot(2, 2, 2)
imagesc(diffP)
colorbar
title('diffP')
axis image

subplot(2, 2, 3)
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

