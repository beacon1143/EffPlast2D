function [Keff, Geff] = get_sigma_2D(loadValue, loadType, nGrid, nTimeSteps, nIter, eIter, needCPUcalc)
    
  % PHYSICS
  Lx  = 20.0;                         % physical length
  Ly  = 20.0;                         % physical width
  E0   = 1.0;                         % Young's modulus
  nu0  = 0.25;                        % Poisson's ratio  
  rho0 = 1.0;                         % density
  K0   = 1.0;  % bulk modulus
  G0   = 0.5;      % shear modulus
  coh  = 0.01;
  P0 = 0.0; %1.0 * coh;
  rad = 1.5;

  % NUMERICS
  %nGrid = 7;
  Nx  = 32 * nGrid;     % number of space steps
  Ny  = 32 * nGrid;
  %Nt  = 10;     % number of time steps
  %nIter = 500;
  %eIter = 1.0e-11;
  CFL = 0.125;     % Courant-Friedrichs-Lewy

  % PREPROCESSING
  dX     = Lx / (Nx - 1);                                   % space step
  dY     = Ly / (Ny - 1);
  x      = (-Lx / 2) : dX : (Lx / 2);                       % space discretization
  y      = (-Ly / 2) : dY : (Ly / 2);
  [x, y] = ndgrid(x, y);                                    % 2D mesh
  xC     = av4(x);
  yC     = av4(y);
  radC   = sqrt(xC .* xC + yC .* yC);
  [xUx, yUx] = ndgrid((-(Lx + dX)/2) : dX : ((Lx + dX)/2), (-Ly/2) : dY : (Ly/2));
  [xUy, yUy] = ndgrid((-Lx/2) : dX : (Lx/2), (-(Ly+dY)/2) : dY : ((Ly+dY)/2));
  dt     = CFL * min(dX, dY) / sqrt( (K0 + 4*G0/3) / rho0);    % time step
  dampX   = 4.0 / dt / Nx;
  dampY   = 4.0 / dt / Ny;
  
  % MATERIALS
  K = zeros(Nx, Ny); %E ./ (3.0 * (1 - 2 * nu));             % bulk modulus
  G = zeros(Nx, Ny); %E ./ (2.0 + 2.0 * nu);                 % shear modulus
  K0 = 1.0;
  G0 = 0.01;
  [K, G] = set_mats_2D(Nx, Ny, x, y, rad, K0, G0);     % Young's modulus and Poisson's ratio

  % INITIAL CONDITIONS
  Pinit   = zeros(Nx, Ny);            % initial hydrostatic stress
  P       = zeros(Nx, Ny);
  tauxyAv = zeros(Nx, Ny);
  Pinit(sqrt(x.*x + y.*y) < rad) = P0;    % hydrostatic stress (ball part of tensor)
  Ux    = zeros(Nx + 1, Ny);        % displacement
  Uy    = zeros(Nx, Ny + 1);
  Vx    = zeros(Nx + 1, Ny);        % velocity
  Vy    = zeros(Nx, Ny + 1);
  tauxx = zeros(Nx, Ny);            % deviatoric stress
  tauyy = zeros(Nx, Ny);
  tauxy = zeros(Nx - 1, Ny - 1);
  J2 = zeros(Nx, Ny);
  J2xy = zeros(Nx - 1, Ny - 1);
  Plast = zeros(Nx, Ny);
  PlastXY = zeros(Nx - 1, Ny - 1);
  
  % ANALYTIC SOLUTION
  Sanrr = zeros(Nx, Ny);
  Sanff = zeros(Nx, Ny);
  Snurr = zeros(Nx, Ny);
  Snuff = zeros(Nx, Ny);
  sinsin = y ./ (sqrt(x.*x + y.*y));
  coscos = x ./ (sqrt(x.*x + y.*y));

  % BOUNDARY CONDITIONS
  dUxdx = loadValue * loadType(1);
  dUydy = loadValue * loadType(2);
  dUxdy = loadValue * loadType(3);
  
  S = zeros(nTimeSteps, 3);

  % INPUT FILES
  pa = [dX, dY, dt, K0, G0, rho0, dampX, dampY, coh, rad, K0, G0];
  
  Keff = zeros(nTimeSteps);

  % parameters
  fil = fopen('pa.dat', 'wb');
  fwrite(fil, pa(:), 'double');
  fclose(fil);

  if needCPUcalc
    % CPU CALCULATION
    for it = 1 : nTimeSteps
      Ux = Ux + (dUxdx * xUx + dUxdy * yUx) / nTimeSteps;
      Uy = Uy + (dUydy * yUy) / nTimeSteps;

      error = 0.0;
      
      for iter = 1 : nIter
        % displacement divergence
        divU = diff(Ux,1,1) / dX + diff(Uy,1,2) / dY;
        
        % constitutive equation - Hooke's law
        P     = Pinit - K .* divU;
        tauxx = 2.0 * G .* (diff(Ux,1,1)/dX - divU/3.0);
        tauyy = 2.0 * G .* (diff(Uy,1,2)/dY - divU/3.0);
        tauxy = av4(G) .* (diff(Ux(2:end-1,:), 1, 2)/dY + diff(Uy(:,2:end-1), 1, 1)/dX);
        
        P(sqrt(x.*x + y.*y) < rad) = 0.0;
        tauxx(sqrt(x.*x + y.*y) < rad) = 0.0;
        tauyy(sqrt(x.*x + y.*y) < rad) = 0.0;
        tauxy(radC < rad) = 0.0;
        
        % tauXY for plasticity
        tauxyAv(2:end-1,2:end-1) = av4(tauxy);
        
        tauxyAv(1, 2:end-1) = tauxyAv(2, 2:end-1);
        tauxyAv(end, 2:end-1) = tauxyAv(end-1, 2:end-1);
        tauxyAv(2:end-1, 1) = tauxyAv(2:end-1, 2);
        tauxyAv(2:end-1, end) = tauxyAv(2:end-1, end-1);
        tauxyAv(1, 1) = 0.5 * (tauxyAv(1, 2) + tauxyAv(2, 1));
        tauxyAv(end, 1) = 0.5 * (tauxyAv(end, 2) + tauxyAv(end-1, 1));
        tauxyAv(1, end) = 0.5 * (tauxyAv(2, end) + tauxyAv(1, end-1));
        tauxyAv(end, end) = 0.5 * (tauxyAv(end, end-1) + tauxyAv(end-1, end));
        
        % plasticity
        J2 = sqrt(tauxx .* tauxx + tauyy .* tauyy + 2.0 * tauxyAv .* tauxyAv);    % Tresca criteria
        J2xy = sqrt(av4(tauxx).^2 + av4(tauyy).^2 + 2.0 * tauxy .* tauxy);
        iPlast = find(J2 > coh);
        if length(iPlast) > 0
          tauxx(iPlast) = tauxx(iPlast) .* coh ./ J2(iPlast);
          tauyy(iPlast) = tauyy(iPlast) .* coh ./ J2(iPlast);
          Plast(iPlast) = 1.0;
        end % if
        iPlastXY = find(J2xy > coh);
        if length(iPlastXY) > 0
          tauxy(iPlastXY) = tauxy(iPlastXY) .* coh ./ J2xy(iPlastXY);
          PlastXY(iPlastXY) = 1.0;
        end % if
        
        % motion equation
        dVxdt = diff(-P(:,2:end-1) + tauxx(:,2:end-1), 1, 1)/dX / rho0 + diff(tauxy,1,2)/dY;
        Vx(2:end-1,2:end-1) = Vx(2:end-1,2:end-1) * (1 - dt * dampX) + dVxdt * dt;
        dVydt = diff(-P(2:end-1,:) + tauyy(2:end-1,:), 1, 2)/dY / rho0 + diff(tauxy,1,1)/dX;
        Vy(2:end-1,2:end-1) = Vy(2:end-1,2:end-1) * (1 - dt * dampY) + dVydt * dt;
        
        % displacements
        Ux = Ux + Vx * dt;
        Uy = Uy + Vy * dt;
        
        % exit criteria
        if mod(iter, 1000) == 0
          error = (max(abs(Vx(:))) / Lx + max(abs(Vy(:))) / Ly) * dt / max(abs(loadValue * loadType));
          if error < eIter
            outStr = sprintf("Number of iterations is %d", iter);
            disp(outStr);
            break
          end
        end %if
      end % for
      
      %tauxyAv(2:end-1,2:end-1) = av4(tauxy);
      
      Plast(1:end-1, 1:end-1) = Plast(1:end-1, 1:end-1) + PlastXY;
      Plast(2:end, 1:end-1) = Plast(2:end, 1:end-1) + PlastXY;
      Plast(1:end-1, 2:end) = Plast(1:end-1, 2:end) + PlastXY;
      Plast(2:end, 2:end) = Plast(2:end, 2:end) + PlastXY;
      
      % cylindrical coorditate system
      Sanrr(Plast > 0) = -P0 + sign(loadValue) * 2.0 * coh * log(sqrt(sqrt(x(Plast > 0) .* x(Plast > 0) + y(Plast > 0) .* y(Plast > 0) ))) / sqrt(2);
      Sanrr(sqrt(x.*x + y.*y) < rad) = 0.0;
      Sanff(Plast > 0) = -P0 + sign(loadValue) * 2.0 * coh * (log(sqrt(sqrt(x(Plast > 0) .* x(Plast > 0) + y(Plast > 0) .* y(Plast > 0)))) + 1.0) / sqrt(2);
      Sanff(sqrt(x.*x + y.*y) < rad) = 0.0;
      
      sinsin = y ./ (sqrt(x.*x + y.*y));
      coscos = x ./ (sqrt(x.*x + y.*y));
      
      %Snurr(Plast > 0) = tauxx(Plast > 0) - P(Plast > 0);
      Snurr(Plast > 0) = (tauxx(Plast > 0) - P(Plast > 0)) .* coscos(Plast > 0) .* coscos(Plast > 0) + ...
                          2.0 * tauxyAv(Plast > 0) .* sinsin(Plast > 0) .* coscos(Plast > 0) + ...
                          (tauyy(Plast > 0) - P(Plast > 0)) .* sinsin(Plast > 0) .* sinsin(Plast > 0);
      Snurr(sqrt(x.*x + y.*y) < rad) = 0.0;
      
      %Snuff(Plast > 0) = tauyy(Plast > 0) - P(Plast > 0);
      Snuff(Plast > 0) = (tauxx(Plast > 0) - P(Plast > 0)) .* sinsin(Plast > 0) .* sinsin(Plast > 0) - ...
                          2.0 * tauxyAv(Plast > 0) .* sinsin(Plast > 0) .* coscos(Plast > 0) + ...
                          (tauyy(Plast > 0) - P(Plast > 0)) .* coscos(Plast > 0) .* coscos(Plast > 0);
      Snuff(sqrt(x.*x + y.*y) < rad) = 0.0;
      
    %% POSTPROCESSING
    %  if mod(it, 1) == 0
    %    subplot(2, 2, 1)
    %    pcolor(x, y, Snurr) % diff(Ux, 1, 1)/dX )
    %    title("\sigma_{rr}")
    %    shading flat
    %    colorbar
    %    axis image        % square image
    %    
    %    subplot(2, 2, 3)
    %    pcolor(x, y, Snuff) % diff(Ux, 1, 1)/dX )
    %    title("\sigma_{\phi \phi}")
    %    shading flat
    %    colorbar
    %    axis image        % square image
    %    
    %    subplot(2, 2, 2)
    %    plot(x(Nx/2 + 1:Nx), 0.5 * (Sanrr(Nx/2 + 1:Nx, Ny/2) + Sanrr(Nx/2 + 1:Nx, Ny/2 - 1)), 'g', x(Nx/2 + 1:Nx), 0.5 * (Snurr(Nx/2 + 1:Nx, Ny/2) + Snurr(Nx/2 + 1:Nx, Ny/2 - 1)), 'r')
    %    title("\sigma_{rr}")
    %    
    %    subplot(2, 2, 4)
    %    plot(x(Nx/2 + 1:Nx), 0.5 * (Sanff(Nx/2 + 1:Nx, Ny/2) + Sanff(Nx/2 + 1:Nx, Ny/2 - 1)), 'g', x(Nx/2 + 1:Nx), 0.5 * (Snuff(Nx/2 + 1:Nx, Ny/2) + Snuff(Nx/2 + 1:Nx, Ny/2 - 1)), 'r')
    %    %plot(x, 0.5 * (Sanrr(:, Ny/2) - Sanff(:, Ny/2)), 'g', x, 0.5 * (Snurr(:, Ny/2) - Snuff(:, Ny/2)), 'r')
    %    title("\sigma_{rr} - \sigma_{\phi \phi}")
    %    
    %    drawnow
    %  endif
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
      
      deltaP = mean(tauxx(1, :) - P(1, :)) + mean(tauxx(end, :) - P(end, :)) + ...
               mean(tauxx(:, 1) - P(:, 1)) + mean(tauxx(:, end) - P(:, end)) + ...
               mean(tauyy(1, :) - P(1, :)) + mean(tauyy(end, :) - P(end, :)) + ...
               mean(tauyy(:, 1) - P(:, 1)) + mean(tauyy(:, end) - P(:, end));
      deltaP = deltaP * 0.125 / coh / sqrt(2);
      
      tauInfty = mean(tauxx(1, :) - tauyy(1, :)) + mean(tauxx(end, :) - tauyy(end, :)) + ...
                 mean(tauxx(:, 1) - tauyy(:, 1)) + mean(tauxx(:, end) - tauyy(:, end));
      tauInfty = tauInfty * 0.125 / coh / sqrt(2);
      
      divUeff = loadValue * (loadType(1) + loadType(2));
      
      % for integration over the solid only
      Psolid = P;
      Psolid(sqrt(x.*x + y.*y) < rad) = 0.0;
      
      tauXXsolid = tauxx;
      tauXXsolid(sqrt(x.*x + y.*y) < rad) = 0.0;
      
      tauYYsolid = tauyy;
      tauYYsolid(sqrt(x.*x + y.*y) < rad) = 0.0;
          
      Keff(it) = -mean(Psolid(:)) / (divUeff) / it * nTimeSteps;
      Geff(it, 1) = 0.5 * mean(tauXXsolid(:)) / (loadValue * loadType(1) - divUeff / 3.0) / it * nTimeSteps;
      Geff(it, 2) = 0.5 * mean(tauYYsolid(:)) / (loadValue * loadType(2) - divUeff / 3.0) / it * nTimeSteps;
      %Geff(it, 3) = mean(tauxy(:)) / (loadValue * loadType(1)) / it * nTimeSteps
      
      dR = max(Ux(2:end, end/2));
      dPhi = pi * ((rad + dR) * (rad + dR) - rad * rad) / Lx / Ly;
      KeffPhi = P(1, end/2) / dPhi
      
      Phi = pi * rad * rad / Lx / Ly;
      Kexact = 0.01 / Phi
    end %for
    
    fil = fopen('Pm.dat', 'wb');
    fwrite(fil, P(:), 'double');
    fclose(fil);
    
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
    
    fil = fopen('tauXYm.dat', 'wb');
    fwrite(fil, tauxy(:), 'double');
    fclose(fil);

    %fil = fopen('tauXYc.dat', 'rb');
    %tauXYc = fread(fil, 'double');
    %fclose(fil);
    %tauXYc = reshape(tauXYc, Nx - 1, Ny - 1);

    %diffP = P - Pc;
    %diffUx = Ux - Uxc;
    %diffUy = Uy - Uyc;
    %diffTauXY = tauxy - tauXYc;
    
    fil = fopen('tauXYavm.dat', 'wb');
    fwrite(fil, tauxyAv(:), 'double');
    fclose(fil);
    
    fil = fopen('J2m.dat', 'wb');
    fwrite(fil, J2(:), 'double');
    fclose(fil);

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
  end %if
  
endfunction
