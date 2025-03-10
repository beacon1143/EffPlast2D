GENERAL DESCRIPTION

This code EffPlast2D calculates the effective dry bulk modulus K_d,  effective pore bulk modulus K_phi, and effective shear modulus G for a preloaded porous elastoplactic material in the 2D case. The method for effective properties estimation is the numerical solution of static elastoplastic problems on a representative volume element (RVE). The finite difference method (FDM) with a regular Cartesian grid and the dynamic relaxation (pseudo-transient) method with an explicit time scheme is used. An RVE in the form of a rectangle, containing a regular arrangement of identical cylindrical pores, is considered for the calculations.

The code is written in MATLAB and C++/CUDA. Input parameters for calculation are hard-coded in MATLAB files. The numerical estimation of the effective moduli is performed using GPU calculations of static elastoplastic problems on an RVE. For the quite small grid resolution, numerical calculations can also be performed on the CPU using MATLAB code. The C++/CUDA code also calculates analytical solution to the static problem and analytical values of the effective moduli using formulas from the following research articles:
- Yarushina, V.M., Dabrowski, M., Podladchikov, Y.Y., 2010. An analytical benchmark with combined pressure and shear loading for elastoplastic numerical models. Geochem Geophy Geosy 11;
- Yarushina, V.M., Podladchikov, Y.Y., 2015. (De)compaction of porous viscoelastoplastic media: Model formulation. Journal of Geophysical Research Solid Earth 120;
- Yarushina, V.M., Podladchikov, Y.Y., Wang, H., 2020. Model for (de)compaction and porosity waves in porous rocks under shear stresses. Journal of Geophysical Research: Solid Earth 125, e2020JB019683;
- Yakovlev, M.Ya., Yarushina, V.M., Bystrov, I.D., Nikitin, L.S., Podladchikov, Y.Y. Effective moduli of porous elastoplastic solids: micromechanical modelling and numerical verification (under publication).
The obtained numerical and analytical values of the effective moduli are printed to the MATLAB command line and to the EffPlast2D.log file. The obtained numerical and analytical solutions of the static problem are exported to the binary *.dat files and then are displayed as MATLAB plots.


INPUT PARAMETERS

Input parameters hard-coded in the eff_props_2D.m file:
1) Lx, Ly - physical dimensions of the RVE (length and width of the rectangle);
2) initLoadValue, loadType - preloading parameters (components of the preloading effective strain tensor are defined as follows: Exx = initLoadValue * loadType(1), Eyy = initLoadValue * loadType(2), Exy = Eyx = initLoadValue * loadType(3));
3) Y - yield stress of the solid material;
4) nPores - number of pores along one side of the RVE (the total number of pores is nPores * nPores);
5) porosity - ratio of the total volume of pores to the volume of the RVE;
6) nTasks - number of static problems to solve (1 for K_d and K_phi calculation in the elastic case, 2 for K_d and K_phi calculation in the elastoplastic case, 3 for K_d, K_phi and G calculation in the elastoplastic case);
7) nGrid - grid parameter (grid resolution is Nx * Ny, where Nx = Ny = 32 * nGrid);
8) nTimeSteps - number of time steps for first static problem (preloading);
9) nIter - maximum number of iterations;
10) eIter - convergence criteria for pseudo-transient calculation (target relative error);
11) iDevice - index of GPU device for the CUDA calculation (0 if the computer has only one NVIDIA graphics card);
12) needCPUcalculation - true if the calculation on CPU (using MATLAB code) is needed (to compare CPU and GPU results), otherwise false.
13) needCompareStatic - true if drawing comparative fields of the numerical and analytical solutions to the static problem is needed, otherwise false.
14) needPeriodicBCs - true if drawing results of static calculation only for internal cells of the volume is needed (can be true only if nPores > 2), otherwise false.

Input parameters hardcoded in the get_sigma_2D.m file:
15) rho0 - pseudo-density of the solid material;
16) K0 - bulk modulus of the solid material;
17) G0 - shear modulus of the solid material;
18) CFL - the Courant number (should be smaller than 1, should be reduced if the iterative process does not converge).


CALCULATION RESULTS

The obtained numerical and analytical values of the effective pore bulk modulus K_phi and the effective dry bulk modulus K_d are printed to the MATLAB command line. Depending on the value of nTasks, different analytical formulas are used: for nTasks == 1, elastic formulas are applied for K_phi and K_d; for nTasks == 2, elastoplastic formulas are used; and for nTasks == 3, elastoplastic formulas are applied for K_phi, K_d, and G, with the effective shear modulus G also being printed.

If nPores > 2, then, additionally, the same effective moduli calculated for the periodic setting (for internal cells only) are printed to the command line.

The obtained numerical and analytical result fields of the static problem solution are exported to the binary *.dat files and then are displayed as MATLAB plots. If needCompareStatic == false, then the numerical nondimensionalized fields of the first and second stress invariants and the displacement magnitude are shown. If needCompareStatic == true, then, additionally, the analytical fields for the same values are shown together with the relative error fields.


HARDWARE AND SOFTWARE REQUIREMENTS

To run GPU calculations using EffPlast2D, your computer must have an NVIDIA graphics card installed. MATLAB, GNU Octave or similar software must be installed (at least for pre- and post-processing). C++ compiler supporting C++14 standard and CUDA ToolKit must be installed (for GPU calculation).