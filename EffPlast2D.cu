#include "EffPlast2D.h"

__global__ void ComputeDisp(double* Ux, double* Uy, double* Vx, double* Vy, 
                            const double* const P,
                            const double* const tauXX, const double* const tauYY, const double* const tauXY,
                            const double* const pa,
                            const long int nX, const long int nY) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  const double dX = pa[0], dY = pa[1];
  const double dT = pa[2];
  const double rho = pa[5];
  const double dampX = pa[6], dampY = pa[7];

  // motion equation
  if (i > 0 && i < nX && j > 0 && j < nY - 1) {
    Vx[j * (nX + 1) + i] = Vx[j * (nX + 1) + i] * (1.0 - dT * dampX) + (dT / rho) * ( (
                           -P[j * nX + i] + P[j * nX + i - 1] + tauXX[j * nX + i] - tauXX[j * nX + i - 1]
                           ) / dX + (
                           tauXY[j * (nX - 1) + i - 1] - tauXY[(j - 1) * (nX - 1) + i - 1]
                           ) / dY );
  }
  if (i > 0 && i < nX - 1 && j > 0 && j < nY) {
    Vy[j * nX + i] = Vy[j * nX + i] * (1.0 - dT * dampY) + (dT / rho) * ( (
                     -P[j * nX + i] + P[(j - 1) * nX + i] + tauYY[j * nX + i] - tauYY[(j - 1) * nX + i]
                     ) / dY + (
                     tauXY[(j - 1) * (nX - 1) + i] - tauXY[(j - 1) * (nX - 1) + i - 1]
                     ) / dX );
  }

  Ux[j * (nX + 1) + i] = Ux[j * (nX + 1) + i] + Vx[j * (nX + 1) + i] * dT;
  Uy[j * nX + i] = Uy[j * nX + i] + Vy[j * nX + i] * dT;
}

__global__ void ComputeStress(const double* const Ux, const double* const Uy,
                              const double* const K, const double* const G,
                              const double* const P0, double* P,
                              double* tauXX, double* tauYY, double* tauXY,
                              const double* const pa,
                              const long int nX, const long int nY) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  const double dX = pa[0], dY = pa[1];
  // const double dT = pa[2];
  const double rad = pa[9];

  // constitutive equation - Hooke's law
  P[j * nX + i] = P0[j * nX + i] - K[j * nX + i] * ( 
                  (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY    // divU
                  );

  /*P[j * nX + i] = P[j * nX + i] - G[j * nX + i] * ( // incompressibility
                  (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY    // divU
                  ) * dT / nX;*/

  tauXX[j * nX + i] = 2.0 * G[j * nX + i] * (
                      (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX -    // dUx/dx
                      ( (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY ) / 3.0    // divU / 3.0
                      );
  tauYY[j * nX + i] = 2.0 * G[j * nX + i] * (
                      (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY -    // dUy/dy
                      ( (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY ) / 3.0    // divU / 3.0
                      );

  if (i < nX - 1 && j < nY - 1) {
    tauXY[j * (nX - 1) + i] = 0.25 * (G[j * nX + i] + G[j * nX + i + 1] + G[(j + 1) * nX + i] + G[(j + 1) * nX + i + 1]) * (
                              (Ux[(j + 1) * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i + 1]) / dY + (Uy[(j + 1) * nX + i + 1] - Uy[(j + 1) * nX + i]) / dX    // dUx/dy + dUy/dx
                              );
  }

  if (sqrt((-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j)) < rad ) {
    P[j * nX + i] = 0.0;
    tauXX[j * nX + i] = 0.0;
    tauYY[j * nX + i] = 0.0;
  }

  if (i < nX - 1 && j < nY - 1) {
    if (sqrt((-0.5 * dX * (nX - 2) + dX * i) * (-0.5 * dX * (nX - 2) + dX * i) + (-0.5 * dY * (nY - 2) + dY * j) * (-0.5 * dY * (nY - 2) + dY * j)) < rad ) {
      tauXY[j * (nX - 1) + i] = 0.0;
    }
  }
}

__global__ void ComputePlasticity(double* tauXX, double* tauYY, double* tauXY,
                                  double* const tauXYav,
                                  double* const J2, double* const J2XY,
                                  const double* const pa,
                                  const long int nX, const long int nY) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  const double coh = pa[8];

  // tauXY for plasticity
  if (i > 0 && i < nX - 1 && 
      j > 0 && j < nY - 1) {
    tauXYav[j * nX + i] = 0.25 * (tauXY[(j - 1) * (nX - 1) + i - 1] + tauXY[(j - 1) * (nX - 1) + i] + tauXY[j * (nX - 1) + i - 1] + tauXY[j * (nX - 1) + i]);
  }
  else if (i == 0 && j > 0 && j < nY - 1) {
    tauXYav[j * nX + i] = 0.25 * (tauXY[(j - 1) * (nX - 1) + i] + tauXY[(j - 1) * (nX - 1) + i + 1] + tauXY[j * (nX - 1) + i] + tauXY[j * (nX - 1) + i + 1]);
  }
  else if (i == nX - 1 && j > 0 && j < nY - 1) {
    tauXYav[j * nX + i] = 0.25 * (tauXY[(j - 1) * (nX - 1) + i - 2] + tauXY[(j - 1) * (nX - 1) + i - 1] + tauXY[j * (nX - 1) + i - 2] + tauXY[j * (nX - 1) + i - 1]);
  }
  else if (i > 0 && i < nX - 1 && j == 0) {
    tauXYav[j * nX + i] = 0.25 * (tauXY[j * (nX - 1) + i - 1] + tauXY[j * (nX - 1) + i] + tauXY[(j + 1) * (nX - 1) + i - 1] + tauXY[(j + 1) * (nX - 1) + i]);
  }
  else if (i > 0 && i < nX - 1 && j == nY - 1) {
    tauXYav[j * nX + i] = 0.25 * (tauXY[(j - 2) * (nX - 1) + i - 1] + tauXY[(j - 2) * (nX - 1) + i] + tauXY[(j - 1) * (nX - 1) + i - 1] + tauXY[(j - 1) * (nX - 1) + i]);
  }
  else if (i == 0 && j == 0) {
    tauXYav[j * nX + i] = 0.25 * (tauXY[j * (nX - 1) + i] + tauXY[j * (nX - 1) + i + 1] + tauXY[(j + 1) * (nX - 1) + i] + tauXY[(j + 1) * (nX - 1) + i + 1]);
  }
  else if (i == 0 && j == nY - 1) {
    tauXYav[j * nX + i] = 0.25 * (tauXY[(j - 2) * (nX - 1) + i] + tauXY[(j - 2) * (nX - 1) + i + 1] + tauXY[(j - 1) * (nX - 1) + i] + tauXY[(j - 1) * (nX - 1) + i + 1]);
  }
  else if (i == nX - 1 && j == 0) {
    tauXYav[j * nX + i] = 0.25 * (tauXY[j * (nX - 1) + i - 2] + tauXY[j * (nX - 1) + i - 1] + tauXY[(j + 1) * (nX - 1) + i - 2] + tauXY[(j + 1) * (nX - 1) + i - 1]);
  }
  else if (i == nX - 1 && j == nY - 1) {
    tauXYav[j * nX + i] = 0.25 * (tauXY[(j - 2) * (nX - 1) + i - 2] + tauXY[(j - 2) * (nX - 1) + i - 1] + tauXY[(j - 1) * (nX - 1) + i - 2] + tauXY[(j - 1) * (nX - 1) + i - 1]);
  }

  if (sqrt((-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j)) < rad ) {
    tauXYav[j * nX + i] = 0.0;
  }

  // plasticity
  J2[j * nX + i] = sqrt( tauXX[j * nX + i] * tauXX[j * nX + i] + tauYY[j * nX + i] * tauYY[j * nX + i] + 2.0 * tauXYav[j * nX + i] * tauXYav[j * nX + i] );
  if (i < nX - 1 && j < nY - 1) {
    J2XY[j * (nX - 1) + i] = sqrt(
      0.0625 * (tauXX[j * nX + i] + tauXX[j * nX + i + 1] + tauXX[(j + 1) * nX + i] + tauXX[(j + 1) * nX + i + 1]) * (tauXX[j * nX + i] + tauXX[j * nX + i + 1] + tauXX[(j + 1) * nX + i] + tauXX[(j + 1) * nX + i + 1]) + 
      0.0625 * (tauYY[j * nX + i] + tauYY[j * nX + i + 1] + tauYY[(j + 1) * nX + i] + tauYY[(j + 1) * nX + i + 1]) * (tauYY[j * nX + i] + tauYY[j * nX + i + 1] + tauYY[(j + 1) * nX + i] + tauYY[(j + 1) * nX + i + 1]) + 
      2.0 * tauXY[j * (nX - 1) + i] * tauXY[j * (nX - 1) + i]
    );
  }

  if (J2[j * nX + i] > coh) {
    tauXX[j * nX + i] *= coh / J2[j * nX + i];
    tauYY[j * nX + i] *= coh / J2[j * nX + i];
    tauXYav[j * nX + i] *= coh / J2[j * nX + i];
    J2[j * nX + i] = sqrt(tauXX[j * nX + i] * tauXX[j * nX + i] + tauYY[j * nX + i] * tauYY[j * nX + i] + 2.0 * tauXYav[j * nX + i] * tauXYav[j * nX + i]);
  }

  if (i < nX - 1 && j < nY - 1) {
    if (J2XY[j * (nX - 1) + i] > coh) {
      tauXY[j * (nX - 1) + i] *= coh / J2XY[j * (nX - 1) + i];
    }
  }
}

std::vector< std::array<double, 3> > EffPlast2D::ComputeSigma(const double loadValue, const std::array<double, 3>& loadType) {
  /* INPUT DATA */

  // displacement
  const double dUxdx = loadValue * loadType[0];
  const double dUydy = loadValue * loadType[1];
  const double dUxdy = loadValue * loadType[2];

  //std::cout << "Before loop...\n";

  std::vector< std::array<double, 3> > Sigma(NT);
  for (auto& i : Sigma) {
    i = {0.0, 0.0, 0.0};
  }

  /* ACTION LOOP */
  for (int it = 0; it < NT; it++) {
    cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nX + 1; i++) {
      for (int j = 0; j < nY; j++) {
        Ux_cpu[j * (nX + 1) + i] += ((-0.5 * dX * nX + dX * i) * dUxdx + (-0.5 * dY * (nY - 1) + dY * j) * dUxdy) / NT;
      }
    }
    cudaMemcpy(Ux_cuda, Ux_cpu, (nX + 1) * nY * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nX; i++) {
      for (int j = 0; j < nY + 1; j++) {
        Uy_cpu[j * nX + i] += (-0.5 * dY * nY + dY * j) * dUydy / NT;
      }
    }
    cudaMemcpy(Uy_cuda, Uy_cpu, nX * (nY + 1) * sizeof(double), cudaMemcpyHostToDevice);

    double error = 0.0;

    /* ITERATION LOOP */
    for (int iter = 0; iter < NITER; iter++) {
      ComputeStress<<<grid, block>>>(Ux_cuda, Uy_cuda, K_cuda, G_cuda, P0_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, /*tauXYav_cuda, J2_cuda, J2XY_cuda,*/ pa_cuda, nX, nY);
      cudaDeviceSynchronize();    // wait for compute device to finish
      ComputePlasticity<<<grid, block>>>(tauXX_cuda, tauYY_cuda, tauXY_cuda, tauXYav_cuda, J2_cuda, J2XY_cuda, pa_cuda, nX, nY);
      cudaDeviceSynchronize();    // wait for compute device to finish
      //std::cout << "After computing sigma...\n";
      ComputeDisp<<<grid, block>>>(Ux_cuda, Uy_cuda, Vx_cuda, Vy_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
      cudaDeviceSynchronize();    // wait for compute device to finish

      if ((iter + 1) % output_step == 0) {
        cudaMemcpy(Vx_cpu, Vx_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Vy_cpu, Vy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost);
        error = (FindMaxAbs(Vx_cpu, (nX + 1) * nY) / (dX * (nX - 1)) + FindMaxAbs(Vy_cpu, nX * (nY + 1)) / (dY * (nY - 1))) * dT /
          (std::abs(loadValue) * std::max( std::max(std::abs(loadType[0]), std::abs(loadType[1])), std::abs(loadType[2]) ));
        std::cout << "Iteration " << iter + 1 << ": Error is " << error << '\n';
        // log_file << "Iteration " << iter + 1 << ": Error is " << error << '\n';
        if (error < EITER) {
          std::cout << "Number of iterations is " << iter + 1 << '\n';
          log_file << "Number of iterations is " << iter + 1 << '\n';
          break;
        }
        else if (iter == NITER - 1) {
          std::cout << "WARNING: Maximum number of iterations reached!\nError is " << error << '\n';
          log_file << "WARNING: Maximum number of iterations reached!\nError is " << error << '\n';
        }
        // std::cout << "Vx on step " << it << " is " << Vx_cpu[nY/2 * (nX + 1) + nX/2] << std::endl;
        // log_file << "Vx on step " << it << " is " << Vx_cpu[nY/2 * (nX + 1) + nX/2] << std::endl;
      }
    }
    /* AVERAGING */
    cudaMemcpy(P_cpu, P_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tauXX_cpu, tauXX_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tauYY_cpu, tauYY_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tauXY_cpu, tauXY_cuda, (nX - 1) * (nY - 1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(J2_cpu, J2_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost);

    /*for (int i = 0; i < nX; i++) {
      for (int j = 0; j < nY; j++) {
        Sigma[it][0] += tauXX_cpu[j * nX + i] - P_cpu[j * nX + i];
        Sigma[it][1] += tauYY_cpu[j * nX + i] - P_cpu[j * nX + i];
      }
    }
    Sigma[it][0] /= nX * nY;
    Sigma[it][1] /= nX * nY;

    for (int i = 0; i < nX - 1; i++) {
      for (int j = 0; j < nY - 1; j++) {
        Sigma[it][2] += tauXY_cpu[j * (nX - 1) + i];
      }
    }
    Sigma[it][2] /= (nX - 1) * (nY - 1);*/

    // -P_eff
    for (int i = 0; i < nX; i++) {
      for (int j = 0; j < nY; j++) {
        if ( sqrt((-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j)) >= rad ) {
          Sigma[it][0] += - P_cpu[j * nX + i];
        }
        else {
          // std::cout << "In the hole!\n";
          // log_file << "In the hole!\n";
        }
      }
    }
    Sigma[it][0] /= nX * nY;

    // Tau_eff
    for (int i = 0; i < nX; i++) {
      for (int j = 0; j < nY; j++) {
        if ( sqrt((-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j)) >= rad ) {
          Sigma[it][1] += tauXX_cpu[j * nX + i];
          Sigma[it][2] += tauYY_cpu[j * nX + i];
        }
      }
    }
    Sigma[it][1] /= nX * nY;
    Sigma[it][2] /= nX * nY;

    // std::cout << Sigma[it][0] / loadValue << '\t' << Sigma[it][1] / loadValue << '\t' << Sigma[it][2] / loadValue << '\n';
    // log_file << Sigma[it][0] / loadValue << '\t' << Sigma[it][1] / loadValue << '\t' << Sigma[it][2] / loadValue << '\n';

    /* ANALYTIC SOLUTION FOR EFFECTIVE PROPERTIES */
    const double deltaP_honest = GetDeltaP_honest();
    const double deltaP_approx = GetDeltaP_approx(loadValue * loadType[0], loadValue * loadType[1]);
    const double tauInfty_approx = GetTauInfty_approx(loadValue * loadType[0], loadValue * loadType[1]);

    int holeX = static_cast<int>((nX + 1) * 2 * rad / nX / dX);    // approx X-axis index of hole boundary
    std::vector<double> dispX((nX + 1) / 2);
    for (int i = (nX + 1) / 2 - holeX - 1; i < (nX + 1) / 2; i++) {
      dispX[i] = Ux_cpu[(nY / 2) * (nX + 1) + i];
    }

    int holeY = static_cast<int>((nY + 1) * 2 * rad / nY / dY);    // approx Y-axis index of hole boundary
    std::vector<double> dispY((nY + 1) / 2);
    for (int j = (nY + 1) / 2 - holeY - 1; j < (nY + 1) / 2; j++) {
      dispY[j] = Uy_cpu[j * nX + nX / 2];
    }

    /*const double dR = FindMaxAbs(Ux_cpu, (nX + 1) * nY);
    std::cout << "dR = " << dR << '\n';
    log_file << "dR = " << dR << '\n';*/
    const double dRx = -FindMaxAbs(dispX);
    std::cout << "dRx = " << dRx << '\n';
    log_file << "dRx = " << dRx << '\n';
    const double dRy = -FindMaxAbs(dispY);
    std::cout << "dRy = " << dRy << '\n';
    log_file << "dRy = " << dRy << '\n';
    const double Phi0 = 3.1415926 * rad * rad / (dX * (nX - 1) * dY * (nY - 1));
    const double Phi = 3.1415926 * (rad + dRx) * (rad + dRy) / (dX * (nX - 1) * dY * (nY - 1) * (1 + loadValue * loadType[0]) * (1 + loadValue * loadType[1]));
    const double dPhi = std::abs(Phi - Phi0); //3.1415926 * ( std::abs((rad + dRx) * (rad + dRy) - rad * rad) ) / (dX * (nX - 1) * dY * (nY - 1));
    // std::cout << "dPhi = " << dPhi << '\n';
    // log_file << "dPhi = " << dPhi << '\n';

    const double KeffPhi = deltaP_approx / dPhi;
    //const double KeffPhi = deltaP_honest / dPhi;
    
    //std::cout << "deltaP_honest = " << deltaP_honest << '\n';
    //log_file << "deltaP_honest = " << deltaP_honest << '\n';
    std::cout << "deltaP / Y = " << deltaP_approx / (pa_cpu[8]) << '\n';
    log_file << "deltaP / Y = " << deltaP_approx / (pa_cpu[8]) << '\n';
    std::cout << "tauInfty / Y = " << tauInfty_approx / (pa_cpu[8]) << '\n';
    log_file << "tauInfty / Y = " << tauInfty_approx / (pa_cpu[8]) << '\n';
    std::cout << "KeffPhi = " << KeffPhi << '\n';
    log_file << "KeffPhi = " << KeffPhi << '\n';

    const double phi = 3.1415926 * rad * rad / (dX * (nX - 1) * dY * (nY - 1));
    const double KexactElast = G0 / phi;
    const double KexactPlast = G0 / phi / exp(std::abs(deltaP_approx) / pa_cpu[8] - 1.0) / 
      (1.0 + 5.0 * tauInfty_approx * tauInfty_approx / pa_cpu[8] / pa_cpu[8]);
    //const double KexactPlast = G0 / phi / exp(std::abs(deltaP_honest) / pa_cpu[8] - 1.0);
    std::cout << "KexactElast = " << KexactElast << '\n';
    log_file << "KexactElast = " << KexactElast << '\n';
    std::cout << "KexactPlast = " << KexactPlast << '\n';
    log_file << "KexactPlast = " << KexactPlast << '\n';

    /* ANALYTIC SOLUTION FOR STATICS */
    double* xxx = new double[nX];
    for (int i = 0; i < nX; i++) {
      xxx[i] = -0.5 * dX * (nX - 1) + dX * i;
    }
    SaveVector(xxx, nX, "xxx_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] xxx;

    double* Sanrr = new double[nX];
    for (int i = 0; i < nX; i++) {
      if (std::abs(-0.5 * dX * (nX - 1) + dX * i) < rad) {
        Sanrr[i] = 0.0;
      }
      else {
        double relR = rad / (-0.5 * dX * (nX - 1) + dX * i);
        Sanrr[i] = -deltaP_approx + deltaP_approx * relR * relR - tauInfty_approx * (1.0 - 4.0 * relR * relR + 3.0 * pow(relR, 4.0));
        /*if (J2_cpu[nY * nX / 2 + i] < (1.0 - std::numeric_limits<double>::epsilon()) * pa_cpu[8]) {
          Sanrr[i] = 0.0;
        }
        else {
          Sanrr[i] = -sqrt(2.0) * pa_cpu[8] * log(1.0 / relR);
        }*/
      }
    }
    
    SaveVector(Sanrr, nX, "Sanrr_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Sanrr;

    double* Sanff = new double[nX];
    for (int i = 0; i < nX; i++) {
      if (std::abs(-0.5 * dX * (nX - 1) + dX * i) < rad) {
        Sanff[i] = 0.0;
      }
      else {
        double relR = rad / (-0.5 * dX * (nX - 1) + dX * i);
        Sanff[i] = -deltaP_approx - deltaP_approx * relR * relR + tauInfty_approx * (1.0 + 3.0 * pow(relR, 4.0));
        /*if (J2_cpu[nY * nX / 2 + i] < (1.0 - std::numeric_limits<double>::epsilon()) * pa_cpu[8]) {
          Sanff[i] = 0.0;
        }
        else {
          Sanff[i] = -sqrt(2.0) * pa_cpu[8] * (1.0 + log(1.0 / relR));
        }*/
      }
    }
    SaveVector(Sanff, nX, "Sanff_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Sanff;

    double* Snurr = new double[nX];
    for (int i = 0; i < nX; i++) {
      Snurr[i] = -P_cpu[nY * nX / 2 + i] + tauXX_cpu[nY * nX / 2 + i];
      // std::cout << Snurr[i] << '\n';
    }
    SaveVector(Snurr, nX, "Snurr_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Snurr;

    double* Snuff = new double[nX];
    for (int i = 0; i < nX; i++) {
      Snuff[i] = -P_cpu[nY * nX / 2 + i] + tauYY_cpu[nY * nX / 2 + i];
    }
    SaveVector(Snuff, nX, "Snuff_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Snuff;
  }

  /* OUTPUT DATA WRITING */
  SaveMatrix(P_cpu, P_cuda, nX, nY, "Pc_" + std::to_string(32 * NGRID) + "_.dat");
  SaveMatrix(tauXX_cpu, tauXX_cuda, nX, nY, "tauXXc_" + std::to_string(32 * NGRID) + "_.dat");
  SaveMatrix(tauYY_cpu, tauYY_cuda, nX, nY, "tauYYc_" + std::to_string(32 * NGRID) + "_.dat");
  SaveMatrix(tauXY_cpu, tauXY_cuda, nX - 1, nY - 1, "tauXYc_" + std::to_string(32 * NGRID) + "_.dat");
  SaveMatrix(tauXYav_cpu, tauXYav_cuda, nX, nY, "tauXYavc_" + std::to_string(32 * NGRID) + "_.dat");
  SaveMatrix(J2_cpu, J2_cuda, nX, nY, "J2c_" + std::to_string(32 * NGRID) + "_.dat");
  SaveMatrix(Ux_cpu, Ux_cuda, nX + 1, nY, "Uxc_" + std::to_string(32 * NGRID) + "_.dat");
  SaveMatrix(Uy_cpu, Uy_cuda, nX, nY + 1, "Uyc_" + std::to_string(32 * NGRID) + "_.dat");

  cudaDeviceReset();
  return Sigma;
}

void EffPlast2D::ReadParams(const std::string& filename) {
  FILE* pa_fil = fopen(filename.c_str(), "rb");
  if (!pa_fil) {
    std::cerr << "Error! Cannot open file pa.dat!\n";
    exit(1);
  }
  fread(pa_cpu, sizeof(double), NPARS, pa_fil);
  fclose(pa_fil);
  cudaMemcpy(pa_cuda, pa_cpu, NPARS * sizeof(double), cudaMemcpyHostToDevice);
}

void EffPlast2D::SetMaterials() {
  //constexpr double K0 = 10.0;
  //constexpr double G0 = 0.01;

  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY; j++) {
      K_cpu[j * nX + i] = K0;
      G_cpu[j * nX + i] = G0;
      if ( sqrt((-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j)) < rad ) {
        K_cpu[j * nX + i] = 0.01 * K0;
        G_cpu[j * nX + i] = 0.01 * G0;
      }
    }
  }

  cudaMemcpy(K_cuda, K_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(G_cuda, G_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);
}

void EffPlast2D::SetInitPressure(const double coh) {
  const double P0 = 0.0; //1.0 * coh;

  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY; j++) {
      P0_cpu[j * nX + i] = 0.0;
      if ( sqrt((-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j)) < rad ) {
        P0_cpu[j * nX + i] = P0;
      }
    }
  }

  cudaMemcpy(P0_cuda, P0_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);
}

void EffPlast2D::SetMatrixZero(double** A_cpu, double** A_cuda, const int m, const int n) {
  *A_cpu = (double*)malloc(m * n * sizeof(double));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      (*A_cpu)[j * m + i] = 0.0;
    }
  }
  cudaMalloc(A_cuda, m * n * sizeof(double));
  cudaMemcpy(*A_cuda, *A_cpu, m * n * sizeof(double), cudaMemcpyHostToDevice);
}

void EffPlast2D::SaveMatrix(double* const A_cpu, const double* const A_cuda, const int m, const int n, const std::string& filename) {
  cudaMemcpy(A_cpu, A_cuda, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  FILE* A_filw = fopen(filename.c_str(), "wb");
  fwrite(A_cpu, sizeof(double), m * n, A_filw);
  fclose(A_filw);
}

void EffPlast2D::SaveVector(double* const arr, const int size, const std::string& filename) {
  FILE* arr_filw = fopen(filename.c_str(), "wb");
  fwrite(arr, sizeof(double), size, arr_filw);
  fclose(arr_filw);
}

double EffPlast2D::FindMaxAbs(const double* const arr, const int size) {
  double max_el = 0.0;
  for (int i = 0; i < size; i++) {
    if (std::abs(arr[i]) > max_el) {
      max_el = std::abs(arr[i]);
    }
  }
  return max_el;
}

double EffPlast2D::FindMaxAbs(const std::vector<double>& vec) {
  double max_el = 0.0;
  for (auto i : vec) {
    if (std::abs(i) > max_el) {
      max_el = i;
    }
  }
  return max_el;
}

double EffPlast2D::GetDeltaP_honest() {
  double deltaP = 0.0, deltaPx = 0.0, deltaPy = 0.0;

  for (int i = 1; i < nX - 1; i++) {
    deltaPx += tauXX_cpu[0 * nX + i] - P_cpu[0 * nX + i];
    deltaPx += tauYY_cpu[0 * nX + i] - P_cpu[0 * nX + i];
    deltaPx += tauXX_cpu[(nY - 1) * nX + i] - P_cpu[(nY - 1) * nX + i];
    deltaPx += tauYY_cpu[(nY - 1) * nX + i] - P_cpu[(nY - 1) * nX + i];
  }
  deltaPx /= (nX - 2);

  for (int j = 1; j < nY - 1; j++) {
    deltaPy += tauXX_cpu[j * nX + 0] - P_cpu[j * nX + 0];
    deltaPy += tauYY_cpu[j * nX + 0] - P_cpu[j * nX + 0];
    deltaPy += tauXX_cpu[j * nX + nY - 1] - P_cpu[j * nX + nY - 1];
    deltaPy += tauYY_cpu[j * nX + nY - 1] - P_cpu[j * nX + nY - 1];
  }
  deltaPy /= (nY - 2);

  deltaP = -0.125 * (deltaPx + deltaPy);
  return deltaP;
}

double EffPlast2D::GetDeltaP_approx(const double Exx, const double Eyy) {
  double deltaP = 0.0;

  if (Exx < Eyy ) {
    deltaP += tauXX_cpu[(nY/2) * nX + 0] - P_cpu[(nY/2) * nX + 0];
    deltaP += tauYY_cpu[(nY/2) * nX + 0] - P_cpu[(nY/2) * nX + 0];
    deltaP += tauXX_cpu[(nY/2) * nX + nX - 1] - P_cpu[(nY/2) * nX + nX - 1];
    deltaP += tauYY_cpu[(nY/2) * nX + nX - 1] - P_cpu[(nY/2) * nX + nX - 1];
  }
  else {
    deltaP += tauXX_cpu[0 * nX + nX/2] - P_cpu[0 * nX + nX/2];
    deltaP += tauYY_cpu[0 * nX + nX/2] - P_cpu[0 * nX + nX/2];
    deltaP += tauXX_cpu[(nY - 1) * nX + nX/2] - P_cpu[(nY - 1) * nX + nX/2];
    deltaP += tauYY_cpu[(nY - 1) * nX + nX/2] - P_cpu[(nY - 1) * nX + nX/2];
  }

  deltaP *= -0.25;
  return deltaP;
}

double EffPlast2D::GetTauInfty_approx(const double Exx, const double Eyy) {
  double tauInfty = 0.0;

  if (Exx < Eyy ) {
    tauInfty += tauYY_cpu[(nY/2) * nX + 0] - tauXX_cpu[(nY/2) * nX + 0];
    tauInfty += tauYY_cpu[(nY/2) * nX + nX - 1] - tauXX_cpu[(nY/2) * nX + nX - 1];
  }
  else {
    tauInfty += tauYY_cpu[0 * nX + nX/2] - tauXX_cpu[0 * nX + nX/2];
    tauInfty += tauYY_cpu[(nY - 1) * nX + nX/2] - tauXX_cpu[(nY - 1) * nX + nX/2];
  }

  tauInfty *= 0.25;
  return tauInfty;
}

EffPlast2D::EffPlast2D() {
  block.x = 32; 
  block.y = 32; 
  grid.x = NGRID;
  grid.y = NGRID;

  nX = block.x * grid.x;
  nY = block.y * grid.y;

  cudaSetDevice(0);
  cudaDeviceReset();
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  /* PARAMETERS */
  pa_cpu = (double*)malloc(NPARS * sizeof(double));
  cudaMalloc((void**)&pa_cuda, NPARS * sizeof(double));
  ReadParams("pa.dat");

  dX = pa_cpu[0];
  dY = pa_cpu[1];
  dT = pa_cpu[2];
  K0 = pa_cpu[3];
  G0 = pa_cpu[4];
  rad = pa_cpu[9];

  /* SPACE ARRAYS */
  // materials
  K_cpu = (double*)malloc(nX * nY * sizeof(double));
  G_cpu = (double*)malloc(nX * nY * sizeof(double));
  cudaMalloc(&K_cuda, nX * nY * sizeof(double));
  cudaMalloc(&G_cuda, nX * nY * sizeof(double));
  SetMaterials();

  // stress
  P0_cpu = (double*)malloc(nX * nY * sizeof(double));
  cudaMalloc(&P0_cuda, nX * nY * sizeof(double));
  SetInitPressure(pa_cpu[8]);

  SetMatrixZero(&P_cpu, &P_cuda, nX, nY);
  SetMatrixZero(&tauXX_cpu, &tauXX_cuda, nX, nY);
  SetMatrixZero(&tauYY_cpu, &tauYY_cuda, nX, nY);
  SetMatrixZero(&tauXY_cpu, &tauXY_cuda, nX - 1, nY - 1);
  SetMatrixZero(&tauXYav_cpu, &tauXYav_cuda, nX, nY);

  // plasticity
  SetMatrixZero(&J2_cpu, &J2_cuda, nX, nY);
  SetMatrixZero(&J2XY_cpu, &J2XY_cuda, nX - 1, nY - 1);

  // displacement
  SetMatrixZero(&Ux_cpu, &Ux_cuda, nX + 1, nY);
  SetMatrixZero(&Uy_cpu, &Uy_cuda, nX, nY + 1);

  // velocity
  SetMatrixZero(&Vx_cpu, &Vx_cuda, nX + 1, nY);
  SetMatrixZero(&Vy_cpu, &Vy_cuda, nX, nY + 1);

  /* UTILITIES */
  log_file.open("EffPlast2D.log");
  output_step = 10'000;
}

EffPlast2D::~EffPlast2D() {
  // parameters
  free(pa_cpu);
  cudaFree(pa_cuda);

  // materials
  free(K_cpu);
  free(G_cpu);
  cudaFree(K_cuda);
  cudaFree(G_cuda);

  // stress
  free(P0_cpu);
  free(P_cpu);
  free(tauXX_cpu);
  free(tauYY_cpu);
  free(tauXY_cpu);
  free(tauXYav_cpu);
  cudaFree(P0_cuda);
  cudaFree(P_cuda);
  cudaFree(tauXX_cuda);
  cudaFree(tauYY_cuda);
  cudaFree(tauXY_cuda);
  cudaFree(tauXYav_cuda);

  // plasticity
  free(J2_cpu);
  free(J2XY_cpu);
  cudaFree(J2_cuda);
  cudaFree(J2XY_cuda);

  // displacement
  free(Ux_cpu);
  free(Uy_cpu);
  cudaFree(Ux_cuda);
  cudaFree(Uy_cuda);

  // velocity
  free(Vx_cpu);
  free(Vy_cpu);
  cudaFree(Vx_cuda);
  cudaFree(Vy_cuda);

  // log
  log_file.close();
}