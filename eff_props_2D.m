clear

loadValue = 0.002;
nGrid = 1;
nTimeSteps = 2;
nIter = 10000;

Sxx = get_sigma_2D(loadValue, [1, 0, 0], nGrid, nTimeSteps, nIter);
Syy = get_sigma_2D(loadValue, [0, 1, 0], nGrid, nTimeSteps, nIter);
Sxy = get_sigma_2D(loadValue, [0, 0, 1], nGrid, nTimeSteps, nIter);

C1111 = zeros(nTimeSteps, 1);
C1122 = zeros(nTimeSteps, 1);
C1112 = zeros(nTimeSteps, 1);
C2222 = zeros(nTimeSteps, 1);
C1222 = zeros(nTimeSteps, 1);
C1212 = zeros(nTimeSteps, 1);

for it = 1:nTimeSteps
  C1111(it) = Sxx(it, 1) / loadValue / it * nTimeSteps
  C1122(it) = Sxx(it, 2) / loadValue / it * nTimeSteps
  C1112(it) = Sxx(it, 3) / loadValue / it * nTimeSteps

  C2222(it) = Syy(it, 2) / loadValue / it * nTimeSteps
  C1222(it) = Syy(it, 3) / loadValue / it * nTimeSteps

  C1212(it) = Sxy(it, 3) / loadValue / it * nTimeSteps
endfor

% GPU CALCULATION
system(['nvcc -DNGRID=', int2str(nGrid), ' -DNT=', int2str(nTimeSteps), ' -DNITER=', int2str(nIter), ' -DNPARS=', int2str(7), ' boundary_problem.cu']);
system(['.\a.exe']);