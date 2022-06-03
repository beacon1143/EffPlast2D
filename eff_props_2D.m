clear
figure(1)
clf
colormap jet

loadValue = -0.004;
nGrid = 30;
nTimeSteps = 1;
nIter = 100000;
eIter = 1.0e-10;
needCPUcalculation = false;
needCompareStatic = true;

Nx  = 32 * nGrid;     % number of space steps
Ny  = 32 * nGrid;

Sxx = get_sigma_2D(loadValue, [1.0, 1.0, 0], nGrid, nTimeSteps, nIter, eIter, needCPUcalculation);

% GPU CALCULATION
system(['nvcc -DNGRID=', int2str(nGrid), ' -DNT=', int2str(nTimeSteps), ' -DNITER=', int2str(nIter), ' -DEITER=', num2str(eIter), ' -DNPARS=', int2str(11), ' EffPlast2D.cu main.cu']);
system(['.\a.exe']);

%fil = fopen(strcat('Pc_', int2str(Nx), '_.dat'), 'rb');
%Pc = fread(fil, 'double');
%fclose(fil);
%Pc = reshape(Pc, Nx, Ny);
%Pc = transpose(Pc);
%
%fil = fopen(strcat('tauXXc_', int2str(Nx), '_.dat'), 'rb');
%tauXXc = fread(fil, 'double');
%fclose(fil);
%tauXXc = reshape(tauXXc, Nx, Ny);
%tauXXc = transpose(tauXXc);
%
%fil = fopen(strcat('tauYYc_', int2str(Nx), '_.dat'), 'rb');
%tauYYc = fread(fil, 'double');
%fclose(fil);
%tauYYc = reshape(tauYYc, Nx, Ny);
%tauYYc = transpose(tauYYc);
%
%fil = fopen(strcat('tauXYc_', int2str(Nx), '_.dat'), 'rb');
%tauXYc = fread(fil, 'double');
%fclose(fil);
%tauXYc = reshape(tauXYc, Nx - 1, Ny - 1);
%tauXYc = transpose(tauXYc);
%
%fil = fopen(strcat('tauXYavc_', int2str(Nx), '_.dat'), 'rb');
%tauXYavc = fread(fil, 'double');
%fclose(fil);
%tauXYavc = reshape(tauXYavc, Nx, Ny);
%tauXYavc = transpose(tauXYavc);
%
%fil = fopen(strcat('J2c_', int2str(Nx), '_.dat'), 'rb');
%J2c = fread(fil, 'double');
%fclose(fil);
%J2c = reshape(J2c, Nx, Ny);
%J2c = transpose(J2c);
%
%fil = fopen(strcat('Uxc_', int2str(Nx), '_.dat'), 'rb');
%Uxc = fread(fil, 'double');
%fclose(fil);
%Uxc = reshape(Uxc, Nx + 1, Ny);
%Uxc = transpose(Uxc);
%
%fil = fopen(strcat('Uyc_', int2str(Nx), '_.dat'), 'rb');
%Uyc = fread(fil, 'double');
%fclose(fil);
%Uyc = reshape(Uyc, Nx, Ny + 1);
%Uyc = transpose(Uyc);
%
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
    fil = fopen(strcat('xxx_', int2str(Nx), '_.dat'), 'rb');
    xxx = fread(fil, 'double');
    fclose(fil);
    xxx = reshape(xxx, Nx, 1);
    
    fil = fopen(strcat('Uanr_', int2str(Nx), '_.dat'), 'rb');
    Uanr = fread(fil, 'double');
    fclose(fil);
    Uanr = reshape(Uanr, Nx, 1);
    
    fil = fopen(strcat('Unur_', int2str(Nx), '_.dat'), 'rb');
    Unur = fread(fil, 'double');
    fclose(fil);
    Unur = reshape(Unur, Nx, 1);

%    fil = fopen(strcat('Sanrr_', int2str(Nx), '_.dat'), 'rb');
%    Sanrr = fread(fil, 'double');
%    fclose(fil);
%    Sanrr = reshape(Sanrr, Nx, 1);
%    
%    fil = fopen(strcat('Sanff_', int2str(Nx), '_.dat'), 'rb');
%    Sanff = fread(fil, 'double');
%    fclose(fil);
%    Sanff = reshape(Sanff, Nx, 1);
%
%    fil = fopen(strcat('Snurr_', int2str(Nx), '_.dat'), 'rb');
%    Snurr = fread(fil, 'double');
%    fclose(fil);
%    Snurr = reshape(Snurr, Nx, 1);
%    
%    fil = fopen(strcat('Snuff_', int2str(Nx), '_.dat'), 'rb');
%    Snuff = fread(fil, 'double');
%    fclose(fil);
%    Snuff = reshape(Snuff, Nx, 1);

%    fil = fopen(strcat('tau_cuda_', int2str(Nx), '_.dat'), 'rb');
%    tau = fread(fil, 'double');
%    fclose(fil);
%    tau = reshape(tau, Nx, Ny);
%    
%    fil = fopen(strcat('P_cuda_', int2str(Nx), '_.dat'), 'rb');
%    P = fread(fil, 'double');
%    fclose(fil);
%    P = reshape(P, Nx, Ny);

    fil = fopen(strcat('Sanrr_', int2str(Nx), '_.dat'), 'rb');
    Sanrr = fread(fil, 'double');
    fclose(fil);
    Sanrr = reshape(Sanrr, Nx - 1, Ny - 1);
    
    fil = fopen(strcat('Sanff_', int2str(Nx), '_.dat'), 'rb');
    Sanff = fread(fil, 'double');
    fclose(fil);
    Sanff = reshape(Sanff, Nx - 1, Ny - 1);

    fil = fopen(strcat('Snurr_', int2str(Nx), '_.dat'), 'rb');
    Snurr = fread(fil, 'double');
    fclose(fil);
    Snurr = reshape(Snurr, Nx - 1, Ny - 1);
    
    fil = fopen(strcat('Snuff_', int2str(Nx), '_.dat'), 'rb');
    Snuff = fread(fil, 'double');
    fclose(fil);
    Snuff = reshape(Snuff, Nx - 1, Ny - 1);
    
    fil = fopen(strcat('plast_', int2str(Nx), '_.dat'), 'rb');
    plast = fread(fil, 'double');
    fclose(fil);
    plast = reshape(plast, Nx - 1, Ny - 1);

    % POSTPROCESSING
    subplot(3, 3, 1)
    imagesc(Snurr)
    colorbar
    title('\sigma_{rr} numerical')
    axis image
    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
    
    subplot(3, 3, 2)
    imagesc(Snuff)
    colorbar
    title('\sigma_{\phi \phi} numerical')
    axis image
    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
    
    subplot(3, 3, 4)
    imagesc(Sanrr)
    colorbar
    title('\sigma_{rr} analytics')
    axis image
    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
    
    subplot(3, 3, 5)
    imagesc(Sanff)
    colorbar
    title('\sigma_{\phi \phi} analytics')
    axis image
    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
    
    eps = 10e-18;
    
    errorSrr =  zeros(Nx - 1, Ny - 1);
    errorSff =  zeros(Nx - 1, Ny - 1);
    
    for i = 1: (Nx - 1)
      for j = 1: (Ny - 1)

        if abs(Sanrr(j, i)) > eps
          errorSrr(j, i) = abs((Snurr(j, i) - Sanrr(j, i)) / Sanrr(j, i));
        end %if
        
        if abs(Sanff(j, i)) > eps
          errorSff(j, i) = abs((Snuff(j, i) - Sanff(j, i)) / Sanff(j, i));
        end %if
        
%        if errorSrr(j, i) > 0.1
%          errorSrr(j, i) = 0.1;
%        end %if
%        
%        if errorSff(j, i) > 0.1
%          errorSff(j, i) = 0.1;
%        end %if
        
      end %for
    end %for
    
    subplot(3, 3, 7)
    imagesc(errorSrr)
    colorbar
    title('\sigma_{rr} error')
    axis image
    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
    
    subplot(3, 3, 8)
    imagesc(errorSff)
    colorbar
    title('\sigma_{\phi \phi} error')
    axis image
    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
    
    subplot(3, 3, 9)
    imagesc(plast)
    colorbar
    title('plast zone')
    axis image
    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
    
%    subplot(1, 5, 3)
%    plot(xxx(Nx/2 + 1:Nx), Sanrr(Nx/2 + 1:Nx), 'LineWidth' , 2, 'g', xxx(Nx/2 + 1:Nx), Snurr(Nx/2 + 1:Nx), 'LineWidth', 2, 'r') 
%    title('\sigma_{rr}')
%    xlabel('r')
%    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
%    %set(findall(gcf,'type','text'),'FontSize',30,'fontWeight','bold')
%    
%    subplot(1, 5, 4)
%    plot(xxx(Nx/2 + 1:Nx), Sanff(Nx/2 + 1:Nx), 'LineWidth' , 2, 'g', xxx(Nx/2 + 1:Nx), Snuff(Nx/2 + 1:Nx), 'LineWidth' , 2, 'r') 
%    title('\sigma_{\phi \phi}')
%    xlabel('r')
%    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
%    
%    subplot(1, 5, 5)
%    plot(xxx(Nx/2 + 1:Nx), Uanr(Nx/2 + 1:Nx), 'LineWidth' , 2, 'g', xxx(Nx/2 + 1:Nx), Unur(Nx/2 + 1:Nx), 'LineWidth' , 2, 'r') 
%    title('U_r')
%    xlabel('r')
%    set(gca, 'FontSize', 15, 'fontWeight', 'bold')
    
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