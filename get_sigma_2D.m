function S = get_sigma_2D(loadValue, loadType, nGrid, nTimeSteps, nIter)
  figure(1)
  clf
  colormap jet

  % PHYSICS
  Lx  = 10.0;                         % physical length
  Ly  = 10.0;                         % physical width
  E0   = 1.0;                      % Young's modulus
  nu0  = 0.25;                     % Poisson's ratio  
  rho0 = 1.0;                      % density
  K0   = E0 / (3.0 * (1 - 2 * nu0));  % bulk modulus
  G0   = E0 / (2.0 + 2.0 * nu0);      % shear modulus

  % NUMERICS
  %nGrid = 1;
  Nx  = 32 * nGrid;     % number of space steps
  Ny  = 32 * nGrid;
  %Nt  = 100000;     % number of time steps
  %nIter = 1000;
  CFL = 0.5;     % Courant-Friedrichs-Lewy

  % PREPROCESSING
  dX     = Lx / (Nx - 1);                                   % space step
  dY     = Ly / (Ny - 1);
  x      = (-Lx / 2) : dX : (Lx / 2);                       % space discretization
  y      = (-Ly / 2) : dY : (Ly / 2);
  [x, y] = ndgrid(x, y);                                    % 2D mesh
  [xUx, yUx] = ndgrid((-(Lx + dX)/2) : dX : ((Lx + dX)/2), (-Ly/2) : dY : (Ly/2));
  [xUy, yUy] = ndgrid((-Lx/2) : dX : (Lx/2), (-(Ly+dY)/2) : dY : ((Ly+dY)/2));
  dt     = CFL * min(dX, dY) / sqrt( (K0 + 4*G0/3) / rho0);    % time step
  damp   = 4.0 / dt / Nx;
  
  % MATERIALS
  E = zeros(Nx, Ny);
  nu = zeros(Nx, Ny);
  [E, nu] = set_mats_2D(Nx, Ny, x, y);     % Young's modulus and Poisson's ratio
  K = E ./ (3.0 * (1 - 2 * nu));             % bulk modulus
  G = E ./ (2.0 + 2.0 * nu);                 % shear modulus

  % INITIAL CONDITIONS
  P0    = zeros(Nx, Ny);            % initial hydrostatic stress
  %P0    = exp(-x .* x - y .* y);    % hydrostatic stress (ball part of tensor)
  Ux    = zeros(Nx + 1, Ny);        % displacement
  Uy    = zeros(Nx, Ny + 1);
  Vx    = zeros(Nx + 1, Ny);        % velocity
  Vy    = zeros(Nx, Ny + 1);
  tauxx = zeros(Nx, Ny);            % deviatoric stress
  tauyy = zeros(Nx, Ny);
  tauxy = zeros(Nx - 1, Ny - 1);

  % BOUNDARY CONDITIONS
  dUxdx = loadValue * loadType(1);
  dUydy = loadValue * loadType(2);
  dUxdy = loadValue * loadType(3);
  
  S = zeros(nTimeSteps, 3);

  % INPUT FILES
  pa = [dX, dY, dt, K0, G0, rho0, damp];

  % parameters
  fil = fopen('pa.dat', 'wb');
  fwrite(fil, pa(:), 'double');
  fclose(fil);

  % CPU CALCULATION
  for it = 1 : nTimeSteps
    Ux = Ux + (dUxdx * xUx + dUxdy * yUx) / nTimeSteps;
    Uy = Uy + (dUydy * yUy) / nTimeSteps;
    for iter = 1 : nIter
      % displacement divergence
      divU = diff(Ux,1,1) / dX + diff(Uy,1,2) / dY;
      
      % constitutive equation - Hooke's law
      P     = P0 - K .* divU;
      tauxx = 2.0 * G .* (diff(Ux,1,1)/dX - divU/3.0);
      tauyy = 2.0 * G .* (diff(Uy,1,2)/dY - divU/3.0);
      tauxy = av4(G) .* (diff(Ux(2:end-1,:), 1, 2)/dY + diff(Uy(:,2:end-1), 1, 1)/dX);
      
      % motion equation
      dVxdt = diff(-P(:,2:end-1) + tauxx(:,2:end-1), 1, 1)/dX / rho0 + diff(tauxy,1,2)/dY;
      Vx(2:end-1,2:end-1) = Vx(2:end-1,2:end-1) * (1 - dt * damp) + dVxdt * dt;
      dVydt = diff(-P(2:end-1,:) + tauyy(2:end-1,:), 1, 2)/dY / rho0 + diff(tauxy,1,1)/dX;
      Vy(2:end-1,2:end-1) = Vy(2:end-1,2:end-1) * (1 - dt * damp) + dVydt * dt;
      
      % displacements
      Ux = Ux + Vx * dt;
      Uy = Uy + Vy * dt;
    endfor
    
  % POSTPROCESSING
  %  if mod(it, 2) == 0
  %    subplot(2, 1, 1)
  %    pcolor(x, y, diff(Ux,1,1)/dX)
  %    title(it)
  %    shading flat
  %    colorbar
  %    axis image        % square image
  %    
  %    subplot(2, 1, 2)
  %    pcolor(x, y, diff(Uy,1,2)/dY)
  %    title(it)
  %    shading flat
  %    colorbar
  %    axis image        % square image
  %    
  %    drawnow
  %  endif
    
    S(it, 1) = mean(tauxx(:) - P(:));
    S(it, 2) = mean(tauyy(:) - P(:));
    S(it, 3) = mean(tauxy(:));
  endfor

  %fil = fopen('Pc.dat', 'rb');
  %Pc = fread(fil, 'double');
  %fclose(fil);
  %Pc = reshape(Pc, Nx, Ny);
  
  %fil = fopen('Uxc.dat', 'rb');
  %Uxc = fread(fil, 'double');
  %fclose(fil);
  %Uxc = reshape(Uxc, Nx + 1, Ny);

  %fil = fopen('Uyc.dat', 'rb');
  %Uyc = fread(fil, 'double');
  %fclose(fil);
  %Uyc = reshape(Uyc, Nx, Ny + 1);

  %fil = fopen('tauXYc.dat', 'rb');
  %tauXYc = fread(fil, 'double');
  %fclose(fil);
  %tauXYc = reshape(tauXYc, Nx - 1, Ny - 1);

  %diffP = P - Pc;
  %diffUx = Ux - Uxc;
  %diffUy = Uy - Uyc;
  %diffTauXY = tauxy - tauXYc;

  % POSTPROCESSING
  %subplot(2,2,1)
  %imagesc(Ux)
  %colorbar
  %title('Ux')
  %axis image

  %subplot(2,2,2)
  %imagesc(diffUx)
  %colorbar
  %title('diffUx')
  %axis image

  %subplot(2,2,3)
  %imagesc(tauxy)
  %colorbar
  %title('tauxy')
  %axis image

  %subplot(2,2,4)
  %imagesc(diffTauXY)
  %colorbar
  %title('diffTauXY')
  %axis image
  
endfunction
