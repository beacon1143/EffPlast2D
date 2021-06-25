function [K, G] = set_mats_2D(N, Nx, Ny, Lx, Ly, x, y, rad, K0, G0)
  %K0 = 10.0;
  %G0 = 0.01;
  %E0 = 9.0 * K0 * G0 / (3.0 * K0 + G0);
  %nu0 = 0.5 * (3.0 * K0 - 2.0 * G0) / (3.0 * K0 + G0);
  K = K0 * ones(Nx, Ny);
  G = G0 * ones(Nx, Ny);
  for i = 0 : N - 1
    for j = 0 : N - 1
      K(sqrt((x - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .* (x - 0.5*Lx*(1-1/N) + (Lx/N)*i) + (y - 0.5*Ly*(1-1/N) + (Ly/N)*j) .* (y - 0.5*Ly*(1-1/N) + (Ly/N)*j)) < rad) = 0.01 * K0;
      G(sqrt((x - 0.5*Lx*(1-1/N) + (Lx/N)*i) .* (x - 0.5*Lx*(1-1/N) + (Lx/N)*i) + (y - 0.5*Ly*(1-1/N) + (Ly/N)*j) .* (y - 0.5*Ly*(1-1/N) + (Ly/N)*j)) < rad) = 0.01 * G0;
    end % for
  end % for
end % function
