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
nPores = 1;
porosity = 0.005;
rad = (0.75 * porosity * Lx * Ly * Lz / (pi * nPores ^ 3)) ^ (1 / 3);
nTasks = 2;

% NUMERICS
nGrid = 2;
nTimeSteps = 1;
nIter = 5000000;
eIter = 1.0e-6;
iDevice = 0;

needCPUcalculation = true;

Nx  = 32 * nGrid;     % number of space steps
Ny  = 32 * nGrid;
Nz  = 32 * nGrid;

get_sigma_3D(Lx, Ly, Lz, initLoadValue, loadType, nGrid, nTimeSteps, nIter, eIter, nPores, Y, porosity, needCPUcalculation);

if not(needCPUcalculation)
  
else % needCPUcalculation
  PmXY = read_data_2D('data\PmXY', Nx, Nx, Ny);
  PmXZ = read_data_2D('data\PmXZ', Nx, Nx, Ny);
  PmYZ = read_data_2D('data\PmYZ', Nx, Nx, Ny);
  tauXXm = read_data_2D('data\tauXXm', Nx, Nx, Ny);
  tauYYm = read_data_2D('data\tauYYm', Nx, Nx, Ny);
  tauXYm = read_data_2D('data\tauXYm', Nx, Nx - 1, Ny - 1);
  
  % POSTPROCESSING
  subplot(1, 3, 1)
  imagesc(PmXY)
  colorbar
  title('PmXY')
  axis image
  
  subplot(1, 3, 2)
  imagesc(PmXZ)
  colorbar
  title('PmXZ')
  axis image
  
  subplot(1, 3, 3)
  imagesc(PmYZ)
  colorbar
  title('PmYZ')
  axis image
  
  %subplot(2, 2, 2)
  %imagesc(tauXXm)
  %colorbar
  %title('tauXXm')
  %axis image
  
  %subplot(2, 2, 3)
  %imagesc(tauYYm)
  %colorbar
  %title('tauYYm')
  %axis image
  
  %subplot(2, 2, 4)
  %imagesc(tauXYm)
  %colorbar
  %title('tauXYm')
  %axis image
  
  drawnow
end % if(needCPUcalculation)