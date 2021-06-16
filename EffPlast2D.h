#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include "cuda.h"

// #define NGRID 1
// #define NPARS 8
// #define NT    2
// #define NITER 100000
// #define EITER 1.0e-11

class EffPlast2D {
public:
  std::vector< std::array<double, 3> > ComputeSigma(const double loadValue, const std::array<double, 3>& loadType);

  EffPlast2D();
  ~EffPlast2D();
private:
  dim3 grid, block;
  long int nX, nY;

  // parameters
  double* pa_cuda, * pa_cpu;
  double dX, dY, dT;
  double rad;                                      // radius of hole
  double K0, G0;                                   // bulk modulus and shear modulus

  // space arrays
  double* K_cpu, * K_cuda, * G_cpu, * G_cuda;      // materials
  double* P0_cpu, * P0_cuda, * P_cpu, * P_cuda;    // stress
  double* tauXX_cpu, * tauXX_cuda;
  double* tauYY_cpu, * tauYY_cuda;
  double* tauXY_cpu, * tauXY_cuda;
  double* tauXYav_cpu, * tauXYav_cuda;
  double* J2_cpu, * J2_cuda;                       // plasticity
  double* J2XY_cpu, * J2XY_cuda;
  double* Ux_cpu, * Ux_cuda;                       // displacement
  double* Uy_cpu, * Uy_cuda;
  double* Vx_cpu, * Vx_cuda;                       // velocity
  double* Vy_cpu, * Vy_cuda;

  // utilities
  std::ofstream log_file;
  size_t output_step;

  void ReadParams(const std::string& filename);
  void SetMaterials();
  void SetInitPressure(const double coh);

  static void SetMatrixZero(double** A_cpu, double** A_cuda, const int m, const int n);
  static void SaveMatrix(double* const A_cpu, const double* const A_cuda, const int m, const int n, const std::string& filename);
  static double FindMaxAbs(const double* const arr, const int size);
  static double FindMaxAbs(const std::vector<double>& vec);

  double GetDeltaP_honest();
  double GetDeltaP_approx();
  double GetTauInfty_approx();
};