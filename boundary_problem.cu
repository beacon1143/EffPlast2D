// Wave 2D 
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include <chrono>
#include "cuda.h"

// #define NGRID 1
// #define NPARS 8
// #define NT    2
// #define NITER 100000

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
                              //double* const tauXYav,
                              //double* const J2, double* const J2XY,
                              const double* const pa,
                              const long int nX, const long int nY) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  const double dX = pa[0], dY = pa[1];
  //const double coh = pa[8];

  // constitutive equation - Hooke's law
  P[j * nX + i] = P0[j * nX + i] - K[j * nX + i] * ( 
                  (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY    // divU
                  );

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

void SetMatrixZero(double** A_cpu, double** A_cuda, const int m, const int n) {
  *A_cpu = (double*)malloc(m * n * sizeof(double));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      (*A_cpu)[j * m + i] = 0.0;
    }
  }
  cudaMalloc(A_cuda, m * n * sizeof(double));
  cudaMemcpy(*A_cuda, *A_cpu, m * n * sizeof(double), cudaMemcpyHostToDevice);
}

void SaveMatrix(double* const A_cpu, const double* const A_cuda, const int m, const int n, const std::string filename) {
  cudaMemcpy(A_cpu, A_cuda, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  FILE* A_filw = fopen(filename.c_str(), "wb");
  fwrite(A_cpu, sizeof(double), m * n, A_filw);
  fclose(A_filw);
}

void SetMaterials(double* const K, double* const G, const int m, const int n, const double dX, const double dY) {
  constexpr double K0 = 1.0;
  constexpr double G0 = 1.0;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      K[j * m + i] = K0;
      G[j * m + i] = G0;
      if ( sqrt((-0.5 * dX * (m - 1) + dX * i) * (-0.5 * dX * (m - 1) + dX * i) + (-0.5 * dY * (n - 1) + dY * j) * (-0.5 * dY * (n - 1) + dY * j)) < 1.0 ) {
        K[j * m + i] = 0.01 * K0;
        G[j * m + i] = 0.01 * G0;
      }
    }
  }
}

void SetInitPressure(double* const P, const double coh,
                     const int m, const int n,
                     const double dX, const double dY) {
  const double P0 = 0.5 * coh;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      P[j * m + i] = 0.0;
      if ( sqrt((-0.5 * dX * (m - 1) + dX * i) * (-0.5 * dX * (m - 1) + dX * i) + (-0.5 * dY * (n - 1) + dY * j) * (-0.5 * dY * (n - 1) + dY * j)) < 1.0 ) {
        P[j * m + i] = P0;
      }
    }
  }
}

std::vector< std::array<double, 3> > ComputeSigma(const double loadValue, const std::array<int, 3>& loadType) {
  dim3 grid, block;
  block.x = 32; 
  block.y = 32; 
  grid.x = NGRID;
  grid.y = NGRID;

  const long int nX = block.x * grid.x;
  const long int nY = block.y * grid.y;

  cudaSetDevice(2);
  cudaDeviceReset();
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  /* INPUT DATA */
  // parameters
  double* pa_cuda;
  double* pa_cpu = (double*)malloc(NPARS * sizeof(double));
  //std::ifstream pa_fil("pa.dat", std::ifstream::in | std::ifstream::binary);
  FILE* pa_fil = fopen("pa.dat", "rb");
  if (!pa_fil) {
    std::cerr << "Error! Cannot open file pa.dat!\n";
    exit(1);
  }
  //pa_fil.read(pa_cpu, NPARS * sizeof(double));
  fread(pa_cpu, sizeof(double), NPARS, pa_fil);
  //pa_fil.close();
  fclose(pa_fil);
  cudaMalloc((void**)&pa_cuda, NPARS * sizeof(double));
  cudaMemcpy(pa_cuda, pa_cpu, NPARS * sizeof(double), cudaMemcpyHostToDevice);

  const double dX = pa_cpu[0], dY = pa_cpu[1];

  // materials
  double* K_cpu = (double*)malloc(nX * nY * sizeof(double));
  double* G_cpu = (double*)malloc(nX * nY * sizeof(double));
  SetMaterials(K_cpu, G_cpu, nX, nY, dX, dY);
  double* K_cuda;
  double* G_cuda;
  cudaMalloc(&K_cuda, nX * nY * sizeof(double));
  cudaMalloc(&G_cuda, nX * nY * sizeof(double));
  cudaMemcpy(K_cuda, K_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(G_cuda, G_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);

  // stress
  double* P0_cpu = (double*)malloc(nX * nY * sizeof(double));
  SetInitPressure(P0_cpu, pa_cpu[8], nX, nY, dX, dY);
  double* P0_cuda;
  cudaMalloc(&P0_cuda, nX * nY * sizeof(double));
  cudaMemcpy(P0_cuda, P0_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);

  double* P_cuda;
  double* P_cpu;
  SetMatrixZero(&P_cpu, &P_cuda, nX, nY);

  double* tauXX_cuda;
  double* tauXX_cpu;
  SetMatrixZero(&tauXX_cpu, &tauXX_cuda, nX, nY);

  double* tauYY_cuda;
  double* tauYY_cpu;
  SetMatrixZero(&tauYY_cpu, &tauYY_cuda, nX, nY);

  double* tauXY_cuda;
  double* tauXY_cpu;
  SetMatrixZero(&tauXY_cpu, &tauXY_cuda, nX - 1, nY - 1);

  double* tauXYav_cuda;
  double* tauXYav_cpu;
  SetMatrixZero(&tauXYav_cpu, &tauXYav_cuda, nX, nY);

  double* J2_cuda;
  double* J2_cpu;
  SetMatrixZero(&J2_cpu, &J2_cuda, nX, nY);

  double* J2XY_cuda;
  double* J2XY_cpu;
  SetMatrixZero(&J2XY_cpu, &J2XY_cuda, nX - 1, nY - 1);

  // displacement
  const double dUxdx = loadValue * loadType[0];
  const double dUydy = loadValue * loadType[1];
  const double dUxdy = loadValue * loadType[2];

  double* Ux_cuda;
  double* Ux_cpu;
  SetMatrixZero(&Ux_cpu, &Ux_cuda, nX + 1, nY);

  double* Uy_cuda;
  double* Uy_cpu;
  SetMatrixZero(&Uy_cpu, &Uy_cuda, nX, nY + 1);

  // velocity
  double* Vx_cuda;
  double* Vx_cpu;
  SetMatrixZero(&Vx_cpu, &Vx_cuda, nX + 1, nY);

  double* Vy_cuda;
  double* Vy_cpu;
  SetMatrixZero(&Vy_cpu, &Vy_cuda, nX, nY + 1);

  //std::cout << "Before loop...\n";

  std::vector< std::array<double, 3> > Sigma(NT);
  for (auto& i : Sigma) {
    i = {0.0, 0.0, 0.0};
  }

  /* ACTION LOOP */
  for (int it = 0; it < NT; it++) {
    for (int i = 0; i < nX + 1; i++) {
      for (int j = 0; j < nY; j++) {
        Ux_cpu[j * (nX + 1) + i] += ((-0.5 * dX * nX + dX * i) * dUxdx + (-0.5 * dY * (nY - 1) + dY * j) * dUxdy) / NT;
      }
    }
    cudaMemcpy(Ux_cuda, Ux_cpu, (nX + 1) * nY * sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < nX; i++) {
      for (int j = 0; j < nY + 1; j++) {
        Uy_cpu[j * nX + i] += (-0.5 * dY * nY + dY * j) * dUydy / NT;
      }
    }
    cudaMemcpy(Uy_cuda, Uy_cpu, nX * (nY + 1) * sizeof(double), cudaMemcpyHostToDevice);

    /* ITERATION LOOP */
    for (int iter = 0; iter < NITER; iter++) {
      ComputeStress<<<grid, block>>>(Ux_cuda, Uy_cuda, K_cuda, G_cuda, P0_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, /*tauXYav_cuda, J2_cuda, J2XY_cuda,*/ pa_cuda, nX, nY);
      cudaDeviceSynchronize();    // wait for compute device to finish
      ComputePlasticity<<<grid, block>>>(tauXX_cuda, tauYY_cuda, tauXY_cuda, tauXYav_cuda, J2_cuda, J2XY_cuda, pa_cuda, nX, nY);
      cudaDeviceSynchronize();    // wait for compute device to finish
      //std::cout << "After computing sigma...\n";
      ComputeDisp<<<grid, block>>>(Ux_cuda, Uy_cuda, Vx_cuda, Vy_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
      cudaDeviceSynchronize();    // wait for compute device to finish

      /*cudaMemcpy(Vx_cpu, Vx_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost);
      std::cout << "Vx on step " << it << " is " << Vx_cpu[nY/2 * (nX + 1) + nX/2] << std::endl;*/
    }
    /* AVERAGING */
    cudaMemcpy(P_cpu, P_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tauXX_cpu, tauXX_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tauYY_cpu, tauYY_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tauXY_cpu, tauXY_cuda, (nX - 1) * (nY - 1) * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nX; i++) {
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
    Sigma[it][2] /= (nX - 1) * (nY - 1);

    std::cout << Sigma[it][0] / loadValue << '\t' << Sigma[it][1] / loadValue << '\t' << Sigma[it][2] / loadValue << '\n';
  }

  /* OUTPUT DATA WRITING */
  SaveMatrix(P_cpu, P_cuda, nX, nY, "Pc.dat");
  SaveMatrix(Ux_cpu, Ux_cuda, nX + 1, nY, "Uxc.dat");
  SaveMatrix(Uy_cpu, Uy_cuda, nX, nY + 1, "Uyc.dat");
  SaveMatrix(tauXY_cpu, tauXY_cuda, nX - 1, nY - 1, "tauXYc.dat");
  SaveMatrix(tauXYav_cpu, tauXYav_cuda, nX, nY, "tauXYavc.dat");
  SaveMatrix(J2_cpu, J2_cuda, nX, nY, "J2c.dat");

  free(pa_cpu);
  free(K_cpu);
  free(G_cpu);
  free(P0_cpu);
  free(P_cpu);
  free(tauXX_cpu);
  free(tauYY_cpu);
  free(tauXY_cpu);
  free(tauXYav_cpu);
  free(J2_cpu);
  free(J2XY_cpu);
  free(Ux_cpu);
  free(Uy_cpu);
  free(Vx_cpu);
  free(Vy_cpu);

  cudaFree(pa_cuda);
  cudaFree(K_cuda);
  cudaFree(G_cuda);
  cudaFree(P0_cuda);
  cudaFree(P_cuda);
  cudaFree(tauXX_cuda);
  cudaFree(tauYY_cuda);
  cudaFree(tauXY_cuda);
  cudaFree(tauXYav_cuda);
  cudaFree(J2_cuda);
  cudaFree(J2XY_cuda);
  cudaFree(Ux_cuda);
  cudaFree(Uy_cuda);
  cudaFree(Vx_cuda);
  cudaFree(Vy_cuda);

  cudaDeviceReset();
  return Sigma;
}

int main() {
  const auto start = std::chrono::system_clock::now();

  constexpr double load_value = 0.00075;
  /*const std::vector< std::array<double, 3> > Sxx = ComputeSigma(load_value, {1, 0, 0});
  const std::vector< std::array<double, 3> > Syy = ComputeSigma(load_value, {0, 1, 0});
  const std::vector< std::array<double, 3> > Sxy = ComputeSigma(load_value, {0, 0, 1});

  std::vector<double> C_1111(NT), C_1122(NT), C_1112(NT),
                      C_2222(NT), C_1222(NT),
                      C_1212(NT);
  
  for (int it = 0; it < NT; it++) {
    C_1111[it] = Sxx[it][0] / load_value / (it + 1) * NT;
    C_1122[it] = Sxx[it][1] / load_value / (it + 1) * NT;
    C_1112[it] = Sxx[it][2] / load_value / (it + 1) * NT;

    C_2222[it] = Syy[it][1] / load_value / (it + 1) * NT;
    C_1222[it] = Syy[it][2] / load_value / (it + 1) * NT;

    C_1212[it] = Sxy[it][2] / load_value / (it + 1) * NT;

    std::cout << "C_1111[" << it << "] = " << C_1111[it] << '\n';
    std::cout << "C_1122[" << it << "] = " << C_1122[it] << '\n';
    std::cout << "C_1112[" << it << "] = " << C_1112[it] << '\n';
    std::cout << "C_2222[" << it << "] = " << C_2222[it] << '\n';
    std::cout << "C_1222[" << it << "] = " << C_1222[it] << '\n';
    std::cout << "C_1212[" << it << "] = " << C_1212[it] << '\n';
  }*/

  const std::vector< std::array<double, 3> > S = ComputeSigma(load_value, { 1, 1, 0 });

  const auto end = std::chrono::system_clock::now();
  const int elapsed_sec = static_cast<int>( std::chrono::duration_cast<std::chrono::seconds>(end - start).count() );
  if (elapsed_sec < 60) {
    std::cout << "Calculation time is " << elapsed_sec << " sec\n";
  }
  else {
    const int elapsed_min = elapsed_sec / 60;
    if (elapsed_min < 60) {
      std::cout << "Calculation time is " << elapsed_min << " min " << elapsed_sec % 60 << " sec\n";
    }
    else {
      std::cout << "Calculation time is " << elapsed_min / 60 << " hours " << elapsed_min % 60 << " min " << elapsed_sec % 60 << " sec\n";
    }
  }

  return 0;
}