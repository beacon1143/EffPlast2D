clear
figure(1)
clf
colormap jet

% PHYSICS
Lx  = 20.0;                         % physical length
Ly  = 20.0;                         % physical width
Lz  = 20.0;                         % physical height
initLoadValue = -0.00004;
loadType = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
Y = 0.00001;
nPores = 4;
porosity = 0.005;
rad = (0.75 * porosity * Lx * Ly * Lz / (pi * nPores ^ 3)) ^ (1 / 3);
nTasks = 2;

% NUMERICS
nGrid = 2;
nTimeSteps = 1;
nIter = 5000000;
eIter = 1.0e-9;
iDevice = 0;

needCPUcalculation = true;

Nx  = 32 * nGrid;     % number of space steps
Ny  = 32 * nGrid;
Nz  = 32 * nGrid;

get_sigma_3D(Lx, Ly, Lz, initLoadValue, loadType, nGrid, nTimeSteps, nIter, eIter, nPores, Y, porosity, needCPUcalculation);