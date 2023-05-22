#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include <set>
#include <limits>
#include <algorithm>
#include <complex>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#if !defined(NGRID)
#define NGRID 32
#endif

#if !defined(NPARS)
#define NPARS 11
#endif

#if !defined(NL)
#define NL 2
#endif

#if !defined(NITER)
#define NITER 100'000
#endif

#if !defined(EITER)
#define EITER 1.0e-10
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		std::cout << "[" + std::string(file) + ":" + std::to_string(line) + "] " + "CUDA error: " + std::string(cudaGetErrorString(code));
		exit(-1);
	}
}

#define gpuGetLastError() gpuErrchk(cudaGetLastError())

class EffPlast2D {
public:
	double EffPlast2D::ComputeKphi( double initLoadValue, double loadValue, 
		unsigned int nTimeSteps, const std::array<double, 3>& loadType
	);

	EffPlast2D();
	~EffPlast2D();
private:
	dim3 grid, block;
	long int nX, nY;

	// input parameters
	double* pa_cuda, * pa_cpu;
	double dX, dY, dT;
	double rad;                                      // radius of hole
	double K0, G0;                                   // bulk modulus and shear modulus
	double E0, nu0;                                  // Young's modulus and Poisson's ratio
	double Y;                                        // yield stress
	double nPores;

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
	double* Vx_cpu, * Vx_cpu_old, * Vx_cuda;                       // velocity
	double* Vxdt_cpu, * Vxdt_cuda;
	double* Vy_cpu, * Vy_cpu_old, * Vy_cuda;
	double* Vydt_cpu, * Vydt_cuda;

	// utilities
	std::ofstream log_file;
	size_t output_step;
	double lX, lY;
	double porosity;
	std::set<std::pair<int, int>> empty_spaces;

	// output parameters
	std::array<std::vector<double>, NL> deltaP;
	std::array<std::vector<double>, NL> deltaPper;
	std::array<std::vector<double>, NL> tauInfty;
	std::array<std::vector<double>, NL> dPhi;
	std::array<std::vector<double>, NL> dPhiPer;
	std::array<double, 3> curEffStrain;
	std::array<std::vector<std::array<double, 3>>, NL> epsilon;
	std::array<std::vector<std::array<double, 4>>, NL> sigma;    // sigma_zz is non-zero due to plane strain

	void ComputeEffParams(const size_t step, const double loadStepValue, const std::array<double, 3>& loadType, const size_t nTimeSteps);

	void ReadParams(const std::string& filename);
	void SetMaterials();
	void SetInitPressure(const double coh);

	static void SetMatrixZero(double** A_cpu, double** A_cuda, const int m, const int n);
	static void SaveMatrix(double* const A_cpu, const double* const A_cuda, const int m, const int n, const std::string& filename);
	static void SaveVector(double* const arr, const int size, const std::string& filename);

	static double FindMaxAbs(const double* const arr, const int size);
	static double FindMaxAbs(const std::vector<double>& vec);

	double GetDeltaP_honest();
	double GetDeltaP_periodic();
	double GetDeltaP_approx(const double Exx, const double Eyy);
	double GetTauInfty_honestest();
	double GetTauInfty_honest();
	double GetTauInfty_approx(const double Exx, const double Eyy);
	[[deprecated]] double GetdPhi();

	void getAnalytic(
		double x, double y, double xi, double kappa, double c0,
		double& cosf,
		double& sinf,
		std::complex<double>& zeta,
		std::complex<double>& w,
		std::complex<double>& dw,
		std::complex<double>& wv,
		std::complex<double>& Phi,
		std::complex<double>& Psi
	);
	std::complex<double> getAnalyticUelast(double x, double y, double tauInfty, double xi, double kappa, double c0);
	double getAnalyticUrHydro(double r, double deltaP);
	void getAnalyticJelast(double x, double y, double xi, double kappa, double c0, double& J1, double& J2);
	void getAnalyticJplast(double r, double xi, double& J1, double& J2);
	void SaveAnStatic2D(const double deltaP, const double tauInfty, const std::array<double, 3>& loadType);
};