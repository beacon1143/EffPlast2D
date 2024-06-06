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
        Vx[j * (nX + 1) + i] = Vx[j * (nX + 1) + i] * (1.0 - dT * dampX) + (dT / rho) * ((
            -P[j * nX + i] + P[j * nX + i - 1] + tauXX[j * nX + i] - tauXX[j * nX + i - 1]
            ) / dX + (
                tauXY[j * (nX - 1) + i - 1] - tauXY[(j - 1) * (nX - 1) + i - 1]
                ) / dY);
    }
    if (i > 0 && i < nX - 1 && j > 0 && j < nY) {
        Vy[j * nX + i] = Vy[j * nX + i] * (1.0 - dT * dampY) + (dT / rho) * ((
            -P[j * nX + i] + P[(j - 1) * nX + i] + tauYY[j * nX + i] - tauYY[(j - 1) * nX + i]
            ) / dY + (
                tauXY[(j - 1) * (nX - 1) + i] - tauXY[(j - 1) * (nX - 1) + i - 1]
                ) / dX);
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
    const double N = pa[10];

    // constitutive equation - Hooke's law
    P[j * nX + i] = P0[j * nX + i] - K[j * nX + i] * (
        (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY    // divU
        );

    /*P[j * nX + i] = P[j * nX + i] - G[j * nX + i] * ( // incompressibility
                    (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY    // divU
                    ) * dT / nX;*/

    tauXX[j * nX + i] = 2.0 * G[j * nX + i] * (
        (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX -    // dUx/dx
        ((Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY) / 3.0    // divU / 3.0
        );
    tauYY[j * nX + i] = 2.0 * G[j * nX + i] * (
        (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY -    // dUy/dy
        ((Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY) / 3.0    // divU / 3.0
        );

    if (i < nX - 1 && j < nY - 1) {
        tauXY[j * (nX - 1) + i] = 0.25 * (G[j * nX + i] + G[j * nX + i + 1] + G[(j + 1) * nX + i] + G[(j + 1) * nX + i + 1]) * (
            (Ux[(j + 1) * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i + 1]) / dY + (Uy[(j + 1) * nX + i + 1] - Uy[(j + 1) * nX + i]) / dX    // dUx/dy + dUy/dx
            );
    }

    for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
            if (sqrt((-0.5 * dX * (nX - 1) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * k) *
                (-0.5 * dX * (nX - 1) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * k) +
                (-0.5 * dY * (nY - 1) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * l) *
                (-0.5 * dY * (nY - 1) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * l)) < rad) {
                P[j * nX + i] = 0.0;
                tauXX[j * nX + i] = 0.0;
                tauYY[j * nX + i] = 0.0;
            }

            if (i < nX - 1 && j < nY - 1) {
                if (sqrt((-0.5 * dX * (nX - 2) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * k) *
                    (-0.5 * dX * (nX - 2) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * k) +
                    (-0.5 * dY * (nY - 2) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * l) *
                    (-0.5 * dY * (nY - 2) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * l)) < rad) {
                    tauXY[j * (nX - 1) + i] = 0.0;
                }
            }
        }
    }
}

__global__ void ComputeJ2(double* tauXX, double* tauYY, double* tauXY, 
    double* const tauXYav, 
    double* const J2, double* const J2XY,
    const long int nX, const long int nY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

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

    J2[j * nX + i] = sqrt(tauXX[j * nX + i] * tauXX[j * nX + i] + tauYY[j * nX + i] * tauYY[j * nX + i] + 2.0 * tauXYav[j * nX + i] * tauXYav[j * nX + i]);
    if (i < nX - 1 && j < nY - 1) {
        J2XY[j * (nX - 1) + i] = sqrt(
            0.0625 * (tauXX[j * nX + i] + tauXX[j * nX + i + 1] + tauXX[(j + 1) * nX + i] + tauXX[(j + 1) * nX + i + 1]) * (tauXX[j * nX + i] + tauXX[j * nX + i + 1] + tauXX[(j + 1) * nX + i] + tauXX[(j + 1) * nX + i + 1]) +
            0.0625 * (tauYY[j * nX + i] + tauYY[j * nX + i + 1] + tauYY[(j + 1) * nX + i] + tauYY[(j + 1) * nX + i + 1]) * (tauYY[j * nX + i] + tauYY[j * nX + i + 1] + tauYY[(j + 1) * nX + i] + tauYY[(j + 1) * nX + i + 1]) +
            2.0 * tauXY[j * (nX - 1) + i] * tauXY[j * (nX - 1) + i]
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

    //const double dX = pa[0], dY = pa[1];
    const double coh = pa[8];
    //const double rad = pa[9];

    /*if (sqrt((-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j)) < rad ) {
      tauXYav[j * nX + i] = 0.0;
    }*/

    // plasticity
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

double EffPlast2D::ComputeKphi(const double initLoadValue, [[deprecated]] const double loadValue, 
    const unsigned int nTimeSteps, const std::array<double, 3>& loadType)
{
    const auto start = std::chrono::system_clock::now();

    const double incPercent = 0.005;    // for calculation of effective moduli with plasticity
    std::array<double, 3> sphericalLoadType{0.5 * (loadType[0] + loadType[1]), 0.5 * (loadType[0] + loadType[1]), 0.0};
    std::array<double, 3> deviatoricLoadType{loadType[0] - sphericalLoadType[0], loadType[1] - sphericalLoadType[1], loadType[2] - sphericalLoadType[2]};

    if (NL != 3)
    {
        std::cerr << "Error! Wrong number of loads!\n";
        exit(1);
    }

    // ComputeEffParams(0, initLoadValue, sphericalLoadType, nTimeSteps);
    // ComputeEffParams(1, initLoadValue, deviatoricLoadType, nTimeSteps);
    // ComputeEffParams(2, initLoadValue * incPercent, loadType, 1);

    ComputeEffParams(0, initLoadValue, loadType, nTimeSteps);
    ComputeEffParams(1, initLoadValue * incPercent, sphericalLoadType, 1);
    ComputeEffParams(2, initLoadValue * incPercent, deviatoricLoadType, 1);

    double Kphi, KphiPer, Kd, KdPer, G, Gper;

    if (NL == 3) {
        double Pinc = deltaP[1][0] - deltaP[0][nTimeSteps - 1];
        double phiInc = dPhi[1][0] - dPhi[0][nTimeSteps - 1];
        Kphi = Pinc / phiInc;
        std::cout << "==============\n" << "Kphi = " << Kphi << std::endl;
        log_file << "==============\n" << "Kphi = " << Kphi << std::endl;

        double PperInc = deltaPper[1][0] - deltaPper[0][nTimeSteps - 1];
        double phiPerInc = dPhiPer[1][0] - dPhiPer[0][nTimeSteps - 1];
        KphiPer = PperInc / phiPerInc;
        std::cout << "KphiPer = " << KphiPer << "\n";
        log_file << "KphiPer = " << KphiPer << "\n";

        std::array<double, 4> sigmaInitLoad = sigma[0][nTimeSteps - 1];
        std::array<double, 4> sigmaVolInc = sigma[1][0];
        std::array<double, 4> sigmaDevInc = sigma[2][0];
        std::array<double, 3> epsilonInitLoad = epsilon[0][nTimeSteps - 1];
        std::array<double, 3> epsilonVolInc = epsilon[1][0];
        std::array<double, 3> epsilonDevInc = epsilon[2][0];

        std::array<double, 4> sigmaInitLoadPer = sigmaPer[0][nTimeSteps - 1];
        std::array<double, 4> sigmaVolIncPer = sigmaPer[1][0];
        std::array<double, 4> sigmaDevIncPer = sigmaPer[2][0];
        std::array<double, 3> epsilonInitLoadPer = epsilonPer[0][nTimeSteps - 1];
        std::array<double, 3> epsilonVolIncPer = epsilonPer[1][0];
        std::array<double, 3> epsilonDevIncPer = epsilonPer[2][0];

        //std::cout << "P = " << -0.5 * (sigma[0][nTimeSteps - 1][0] + sigma[0][nTimeSteps - 1][1]) << "\n";
        //std::cout << "divU = " << epsilon[0][nTimeSteps - 1][0] + epsilon[0][nTimeSteps - 1][1] << "\n";
        double pInc = -0.33333333 * (sigmaVolInc[0] + sigmaVolInc[1] + sigmaVolInc[2] - sigmaInitLoad[0] - sigmaInitLoad[1] - sigmaInitLoad[2]);
        double epsInc = epsilonVolInc[0] + epsilonVolInc[1] - epsilonInitLoad[0] - epsilonInitLoad[1];
        Kd = -pInc / epsInc;
        std::cout << "Kd = " << Kd << "\n";
        log_file << "Kd = " << Kd << "\n";

        double pIncPer = -0.33333333 * (sigmaVolIncPer[0] + sigmaVolIncPer[1] + sigmaVolIncPer[2] - sigmaInitLoadPer[0] - sigmaInitLoadPer[1] - sigmaInitLoadPer[2]);
        double epsIncPer = epsilonVolIncPer[0] + epsilonVolIncPer[1] - epsilonInitLoadPer[0] - epsilonInitLoadPer[1];
        KdPer = -pIncPer / epsIncPer;
        std::cout << "KdPer = " << KdPer << "\n";
        log_file << "KdPer = " << KdPer << "\n";
 
        G = 0.5 * (sigmaDevInc[0] - sigmaVolInc[0] - sigmaDevInc[1] + sigmaVolInc[1]) / 
            (epsilonDevInc[0] - epsilonVolInc[0] - epsilonDevInc[1] + epsilonVolInc[1]);
        std::cout << "G = " << G << "\n";
        log_file << "G = " << G << "\n";

        Gper = 0.5 * (sigmaDevIncPer[0] - sigmaVolIncPer[0] - sigmaDevIncPer[1] + sigmaVolIncPer[1]) / 
            (epsilonDevIncPer[0] - epsilonVolIncPer[0] - epsilonDevIncPer[1] + epsilonVolIncPer[1]);
        std::cout << "Gper = " << Gper << "\n";
        log_file << "Gper = " << Gper << "\n";
    }

    if (NL && nTimeSteps) {
        SaveAnStatic2D(deltaP[NL - 2][nTimeSteps - 1], tauInfty[NL - 2][nTimeSteps - 1], loadType);
    }

    /* OUTPUT DATA WRITING */
    SaveMatrix(P_cpu, P_cuda, nX, nY, "data/Pc_" + std::to_string(32 * NGRID) + "_.dat");
    SaveMatrix(tauXX_cpu, tauXX_cuda, nX, nY, "data/tauXXc_" + std::to_string(32 * NGRID) + "_.dat");
    //SaveMatrix(tauYY_cpu, tauYY_cuda, nX, nY, "data/tauYYc_" + std::to_string(32 * NGRID) + "_.dat");
    //SaveMatrix(tauXY_cpu, tauXY_cuda, nX - 1, nY - 1, "data/tauXYc_" + std::to_string(32 * NGRID) + "_.dat");
    //SaveMatrix(tauXYav_cpu, tauXYav_cuda, nX, nY, "data/tauXYavc_" + std::to_string(32 * NGRID) + "_.dat");
    //SaveMatrix(J2_cpu, J2_cuda, nX, nY, "data/J2c_" + std::to_string(32 * NGRID) + "_.dat");
    //SaveMatrix(Ux_cpu, Ux_cuda, nX + 1, nY, "data/Uxc_" + std::to_string(32 * NGRID) + "_.dat");
    //SaveMatrix(Uy_cpu, Uy_cuda, nX, nY + 1, "data/Uyc_" + std::to_string(32 * NGRID) + "_.dat");

    //gpuErrchk(cudaDeviceReset());
    const auto end = std::chrono::system_clock::now();

    int elapsed_sec = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(end - start).count());
    if (elapsed_sec < 60) {
        std::cout << "Calculation time is " << elapsed_sec << " sec\n";
        log_file << "Calculation time is " << elapsed_sec << " sec\n\n\n";
    }
    else {
        int elapsed_min = elapsed_sec / 60;
        elapsed_sec = elapsed_sec % 60;
        if (elapsed_min < 60) {
            std::cout << "Calculation time is " << elapsed_min << " min " << elapsed_sec << " sec\n";
            log_file << "Calculation time is " << elapsed_min << " min " << elapsed_sec << " sec\n\n\n";
        }
        else {
            int elapsed_hour = elapsed_min / 60;
            elapsed_min = elapsed_min % 60;
            if (elapsed_hour < 24) {
                std::cout << "Calculation time is " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n";
                log_file << "Calculation time is " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n\n\n";
            }
            else {
                const int elapsed_day = elapsed_hour / 24;
                elapsed_hour = elapsed_hour % 24;
                if (elapsed_day < 7) {
                    std::cout << "Calculation time is " << elapsed_day << " days " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n";
                    log_file << "Calculation time is " << elapsed_day << " days " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n\n\n";
                }
                else {
                    std::cout << "Calculation time is " << elapsed_day / 7 << " weeks " << elapsed_day % 7 << " days " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n";
                    log_file << "Calculation time is " << elapsed_day / 7 << " weeks " << elapsed_day % 7 << " days " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n\n\n";
                }
            }
        }
    }

    return Kphi;
}

void EffPlast2D::ComputeEffParams(const size_t step, const double loadStepValue, const std::array<double, 3>& loadType, const size_t nTimeSteps) {
    std::cout << "\nLOAD STEP " << step + 1 << "\n";
    log_file << "\nLOAD STEP " << step + 1 << "\n";
    std::cout << "Porosity is " << porosity * 100 << "%\n";
    log_file << "Porosity is " << porosity * 100 << "%\n";
    std::cout << "Mesh resolution is " << nX << "x" << nY << "\n\n";
    log_file << "Mesh resolution is " << nX << "x" << nY << "\n\n";

    deltaP[step].resize(nTimeSteps);
    deltaPper[step].resize(nTimeSteps);
    tauInfty[step].resize(nTimeSteps);
    dPhi[step].resize(nTimeSteps);
    dPhiPer[step].resize(nTimeSteps);
    epsilon[step].resize(nTimeSteps);
    epsilonPer[step].resize(nTimeSteps);
    sigma[step].resize(nTimeSteps);
    sigmaPer[step].resize(nTimeSteps);

    double dUxdx = 0.0;
    double dUydy = 0.0;
    double dUxdy = 0.0;
    double dUydx = 0.0;

    if (step == 0) {
        curEffStrain = { 0.0 };
        memset(Ux_cpu, 0, (nX + 1) * nY * sizeof(double));
        memset(Uy_cpu, 0, nX * (nY + 1) * sizeof(double));
    }
    else { // additional loading
        gpuErrchk(cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost));
    }

    /* ACTION LOOP */
    for (int it = 0; it < nTimeSteps; it++) {
        std::cout << "Time step " << (it + 1) << " from " << nTimeSteps << std::endl;
        log_file << "Time step " << (it + 1) << " from " << nTimeSteps << std::endl;

        epsilon[step][it] = { 0.0 };
        epsilonPer[step][it] = { 0.0 };
        sigma[step][it] = { 0.0 };
        sigmaPer[step][it] = { 0.0 };

        dUxdx = loadStepValue * loadType[0] / static_cast<double>(nTimeSteps);
        dUydy = loadStepValue * loadType[1] / static_cast<double>(nTimeSteps);
        dUxdy = loadStepValue * loadType[2] / static_cast<double>(nTimeSteps);
        dUydx = dUxdy;

        curEffStrain[0] += dUxdx;
        curEffStrain[1] += dUydy;
        curEffStrain[2] += dUxdy;
        epsilon[step][it] = curEffStrain;

        std::cout << "Macro strain: (" << curEffStrain[0] << ", " << curEffStrain[1] << ", " << curEffStrain[2] << ")\n\n";
        log_file << "Macro strain: (" << curEffStrain[0] << ", " << curEffStrain[1] << ", " << curEffStrain[2] << ")\n\n";

        if (it > 0) {    // non-first time step
            gpuErrchk(cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost));
        }

        //std::cout << "Ux = " << Ux_cpu[(3 * nY / 4) * (nX + 1) + 3 * nX / 4] << "\nUy = " << Uy_cpu[(3 * nY / 4) * nX + 3 * nX / 4] << "\n";

        for (int i = 0; i < nX + 1; i++) {
            for (int j = 0; j < nY; j++) {
                Ux_cpu[j * (nX + 1) + i] += (-0.5 * dX * nX + dX * i) * dUxdx + (-0.5 * dY * (nY - 1) + dY * j) * dUxdy;
            }
        }
        gpuErrchk(cudaMemcpy(Ux_cuda, Ux_cpu, (nX + 1) * nY * sizeof(double), cudaMemcpyHostToDevice));
        for (int i = 0; i < nX; i++) {
            for (int j = 0; j < nY + 1; j++) {
                Uy_cpu[j * nX + i] += (-0.5 * dY * nY + dY * j) * dUydy + (-0.5 * dX * (nX - 1) + dX * i) * dUydx;
            }
        }
        gpuErrchk(cudaMemcpy(Uy_cuda, Uy_cpu, nX * (nY + 1) * sizeof(double), cudaMemcpyHostToDevice));

        //std::cout << "dUxdx = " << dUxdx << "\ndUydy = " << dUydy << "\ndUxdy = " << dUxdy << "\n";
        //std::cout << "Ux = " << Ux_cpu[(3 * nY / 4) * (nX + 1) /*+ 3 * nX / 4*/] << "\nUy = " << Uy_cpu[(3 * nY / 4) * nX /*+ 3 * nX / 4*/] << "\n";

        double error = 0.0;

        /* ITERATION LOOP */
        for (int iter = 0; iter < NITER; iter++) {
            ComputeStress<<<grid, block>>>(Ux_cuda, Uy_cuda, K_cuda, G_cuda, P0_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, /*tauXYav_cuda, J2_cuda, J2XY_cuda,*/ pa_cuda, nX, nY);
            gpuErrchk(cudaDeviceSynchronize());
            ComputeJ2<<<grid, block>>>(tauXX_cuda, tauYY_cuda, tauXY_cuda, tauXYav_cuda, J2_cuda, J2XY_cuda, nX, nY);
            gpuErrchk(cudaDeviceSynchronize());
            ComputePlasticity<<<grid, block>>>(tauXX_cuda, tauYY_cuda, tauXY_cuda, tauXYav_cuda, J2_cuda, J2XY_cuda, pa_cuda, nX, nY);
            gpuErrchk(cudaDeviceSynchronize());
            ComputeDisp<<<grid, block>>>(Ux_cuda, Uy_cuda, Vx_cuda, Vy_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
            gpuErrchk(cudaDeviceSynchronize());

            /*if (iter == 1000) {
                gpuErrchk(cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost));
                std::cout << "Ux1 = " << Ux_cpu[(3 * nY / 4) * (nX + 1) + 3 * nX / 4] << "\nUy1 = " << Uy_cpu[(3 * nY / 4) * nX + 3 * nX / 4] << "\n";
            }*/

            if ((iter + 1) % output_step == 0) {
                gpuErrchk(cudaMemcpy(Vx_cpu, Vx_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(Vy_cpu, Vy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost));

                error = (FindMaxAbs(Vx_cpu, (nX + 1) * nY) / (dX * (nX - 1)) + FindMaxAbs(Vy_cpu, nX * (nY + 1)) / (dY * (nY - 1))) * dT /
                    (std::max(std::abs(curEffStrain[0]), std::max(curEffStrain[1], curEffStrain[2])));
                    //(std::abs(loadStepValue) * std::max(std::max(std::abs(loadType[0]), std::abs(loadType[1])), std::abs(loadType[2])));

                std::cout << "Iteration " << iter + 1 << ": Error is " << error << std::endl;
                log_file << "Iteration " << iter + 1 << ": Error is " << error << std::endl;

                if (error < EITER) {
                    std::cout << "Number of iterations is " << iter + 1 << '\n';
                    log_file << "Number of iterations is " << iter + 1 << '\n';
                    break;
                }
                else if (iter == NITER - 1) {
                    std::cout << "WARNING: Maximum number of iterations reached!\nError is " << error << '\n';
                    log_file << "WARNING: Maximum number of iterations reached!\nError is " << error << '\n';
                }
            }
        } // for(iter), iteration loop

        /* AVERAGING */
        gpuErrchk(cudaMemcpy(P_cpu, P_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(tauXX_cpu, tauXX_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(tauYY_cpu, tauYY_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(tauXY_cpu, tauXY_cuda, (nX - 1) * (nY - 1) * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(tauXYav_cpu, tauXYav_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(J2_cpu, J2_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost));

        // auto inTheHole = [&](double X, double Y) -> bool {
        //     for (int k = 0; k < nPores; k++)
        //     {
        //         for (int l = 0; l < nPores; l++)
        //         {
        //             const double cx = 0.5 * dX * (nX - 1) * (1.0 - 1.0 / nPores) - (dX * (nX - 1) / nPores) * k;
        //             const double cy = 0.5 * dY * (nY - 1) * (1.0 - 1.0 / nPores) - (dY * (nY - 1) / nPores) * l;

        //             if ((X - cx) * (X - cx) + (Y - cy) * (Y - cy) < rad * rad)
        //                 return true;
        //         }
        //     }
        //     return false;
        // };

        const int perOffsetX = nX / nPores;
        const int perOffsetY = nY / nPores;

        for (int i = 1; i < nX - 1; i++) {
            for (int j = 1; j < nY - 1; j++) {
                const double x = -0.5 * dX * (nX - 1) + dX * i;
                const double y = -0.5 * dY * (nY - 1) + dY * j;

                //if (!inTheHole(x, y)) 
                sigma[step][it][0] += tauXX_cpu[j * nX + i] - P_cpu[j * nX + i];
                sigma[step][it][1] += tauYY_cpu[j * nX + i] - P_cpu[j * nX + i];
                sigma[step][it][2] += nu0 * (tauXX_cpu[j * nX + i] + tauYY_cpu[j * nX + i] - 2.0 * P_cpu[j * nX + i]);

                if (i < nX - 1 && j < nY - 1)
                    sigma[step][it][3] += tauXY_cpu[j * (nX - 1) + i];

                if (i > perOffsetX && i < nX - perOffsetX && j > perOffsetY && j < nY - perOffsetY)
                {
                    sigmaPer[step][it][0] += tauXX_cpu[j * nX + i] - P_cpu[j * nX + i];
                    sigmaPer[step][it][1] += tauYY_cpu[j * nX + i] - P_cpu[j * nX + i];
                    sigmaPer[step][it][2] += nu0 * (tauXX_cpu[j * nX + i] + tauYY_cpu[j * nX + i] - 2.0 * P_cpu[j * nX + i]);
                    sigmaPer[step][it][3] += 0.25 * (
                        tauXY_cpu[(j - 1) * (nX - 1) + i - 1] + 
                        tauXY_cpu[(j - 1) * (nX - 1) + i] + 
                        tauXY_cpu[j * (nX - 1) + i - 1] + 
                        tauXY_cpu[j * (nX - 1) + i]
                    );

                    epsilonPer[step][it][0] += (Ux_cpu[j * (nX + 1) + i + 1] - Ux_cpu[j * (nX + 1) + i]) / dX;
                    epsilonPer[step][it][1] += (Uy_cpu[(j + 1) * nX + i] - Uy_cpu[j * nX + i]) / dY;
                    epsilonPer[step][it][2] += 0.125 * (
                        (Ux_cpu[j * (nX + 1) + i] - Ux_cpu[(j - 1) * (nX + 1) + i]) / dY + (Uy_cpu[j * nX + i] - Uy_cpu[j * nX + i - 1]) / dX +
                        (Ux_cpu[j * (nX + 1) + i + 1] - Ux_cpu[(j - 1) * (nX + 1) + i + 1]) / dY + (Uy_cpu[j * nX + i + 1] - Uy_cpu[j * nX + i]) / dX +
                        (Ux_cpu[(j + 1) * (nX + 1) + i] - Ux_cpu[j * (nX + 1) + i]) / dY + (Uy_cpu[(j + 1) * nX + i] - Uy_cpu[(j + 1) * nX + i - 1]) / dX +
                        (Ux_cpu[(j + 1) * (nX + 1) + i + 1] - Ux_cpu[j * (nX + 1) + i + 1]) / dY + (Uy_cpu[(j + 1) * nX + i + 1] - Uy_cpu[(j + 1) * nX + i]) / dX
                    );
                }
            }
        }
        sigma[step][it][0] /= nX * nY;
        sigma[step][it][1] /= nX * nY;
        sigma[step][it][2] /= nX * nY;
        sigma[step][it][3] /= (nX - 1) * (nY - 1);

        sigmaPer[step][it][0] /= (nX - 2 * perOffsetX) * (nY - 2 * perOffsetY);
        sigmaPer[step][it][1] /= (nX - 2 * perOffsetX) * (nY - 2 * perOffsetY);
        sigmaPer[step][it][2] /= (nX - 2 * perOffsetX) * (nY - 2 * perOffsetY);
        sigmaPer[step][it][3] /= (nX - 2 * perOffsetX) * (nY - 2 * perOffsetY);

        epsilonPer[step][it][0] /= (nX - 2 * perOffsetX) * (nY - 2 * perOffsetY);
        epsilonPer[step][it][1] /= (nX - 2 * perOffsetX) * (nY - 2 * perOffsetY);
        epsilonPer[step][it][2] /= (nX - 2 * perOffsetX) * (nY - 2 * perOffsetY);

        /* ANALYTIC SOLUTION FOR EFFECTIVE PROPERTIES */
        deltaP[step][it] = GetDeltaP_honest();
        /*std::cout << "deltaP = " << deltaP[step][it] << '\n';
        log_file << "deltaP = " << deltaP[step][it] << '\n';*/
        deltaPper[step][it] = GetDeltaP_periodic();
        /*std::cout << "deltaPper = " << deltaPper[step][it] << '\n';
        log_file << "deltaPper = " << deltaPper[step][it] << '\n';*/
        tauInfty[step][it] = GetTauInfty_honest();

        double Lx = dX * (nX - 1);
        double Ly = dY * (nY - 1);
        // set zero Ux in the pores
        for (int i = 0; i < nX + 1; i++) {
            for (int j = 0; j < nY; j++) {
                double x = -0.5 * dX * nX + dX * i;
                double y = -0.5 * dY * (nY - 1) + dY * j;
                for (int k = 0; k < nPores; k++) {
                    for (int l = 0; l < nPores; l++) {
                        if (sqrt((x - 0.5 * Lx * (1.0 - 1.0 / nPores) + (Lx / nPores) * k) * (x - 0.5 * Lx * (1.0 - 1.0 / nPores) + (Lx / nPores) * k) +
                            (y - 0.5 * Ly * (1.0 - 1.0 / nPores) + (Ly / nPores) * l) * (y - 0.5 * Ly * (1.0 - 1.0 / nPores) + (Ly / nPores) * l)) < rad)
                        {
                            Ux_cpu[j * (nX + 1) + i] = 0.0;
                        }
                    }
                }
            }
        }
        // set zero Ux in the pores
        for (int i = 0; i < nX; i++) {
            for (int j = 0; j < nY + 1; j++) {
                double x = -0.5 * dX * (nX - 1) + dX * i;
                double y = -0.5 * dY * nY + dY * j;
                for (int k = 0; k < nPores; k++) {
                    for (int l = 0; l < nPores; l++) {
                        if (sqrt((x - 0.5 * Lx * (1.0 - 1.0 / nPores) + (Lx / nPores) * k) * (x - 0.5 * Lx * (1.0 - 1.0 / nPores) + (Lx / nPores) * k) +
                            (y - 0.5 * Ly * (1.0 - 1.0 / nPores) + (Ly / nPores) * l) * (y - 0.5 * Ly * (1.0 - 1.0 / nPores) + (Ly / nPores) * l)) < rad)
                        {
                            Uy_cpu[j * nX + i] = 0.0;
                        }
                    }
                }
            }
        }
        
        double HoleAreaPi = 0.0; // HoleArea / Pi
        double InternalHoleAreaPi = 0.0;

        for (int k = 0; k < nPores; k++) {
            for (int l = 0; l < nPores; l++) {
                const double cxdX = 0.5 * (nX - 1) * (1.0 - 1.0 / nPores) - (static_cast<double>(nX - 1) / nPores) * k; // cx / dX
                const double cydY = 0.5 * (nY - 1) * (1.0 - 1.0 / nPores) - (static_cast<double>(nY - 1) / nPores) * l; // cy / dY

                // horizontal displacements
                // left point of a hole
                const size_t cyIdx = static_cast<size_t>(cydY + 0.5 * (nY - 1));
                size_t rxIdx = static_cast<size_t>(cxdX - rad / dX + 0.5 * nX);

                std::vector<double> dispXleft(5);
                for (int i = 0; i < 5; i++) {
                    dispXleft[i] = Ux_cpu[cyIdx * (nX + 1) + rxIdx - 1 + i];
                    //std::cout << "j = " << cyIdx << " i = " << rxIdx - 1 + i << "\n";
                    //std::cout << dispXleft[i] << "\n";
                }

                // right point of a hole
                rxIdx = static_cast<size_t>(cxdX + rad / dX + 0.5 * nX);
                std::vector<double> dispXright(5);
                for (int i = 0; i < 5; i++) {
                    dispXright[i] = Ux_cpu[cyIdx * (nX + 1) + rxIdx - 2 + i];
                    //std::cout << dispXright[i] << "\n";
                }

                // vertical displacements
                // bottom point of a hole
                const size_t cxIdx = static_cast<size_t>(cxdX + 0.5 * (nX - 1));
                size_t ryIdx = static_cast<size_t>(cydY - rad / dY + 0.5 * nY);

                std::vector<double> dispYbottom(5);
                for (int j = 0; j < 5; j++) {
                    dispYbottom[j] = Uy_cpu[(ryIdx - 1 + j) * nX + cxIdx];
                    //std::cout << dispYbottom[j] << "\n";
                }

                // top point of a hole
                ryIdx = static_cast<size_t>(cydY + rad / dY + 0.5 * nY);
                std::vector<double> dispYtop(5);
                for (int j = 0; j < 5; j++) {
                    dispYtop[j] = Uy_cpu[(ryIdx - 2 + j) * nX + cxIdx];
                    //std::cout << dispYtop[j] << "\n";
                }

                //std::cout << "dRxLeft = " << FindMaxAbs(dispXleft) << ", dRxRight = " << FindMaxAbs(dispXright) << "\n";
                const double dRx = -0.5 * (FindMaxAbs(dispXleft) - FindMaxAbs(dispXright));
                const double dRy = -0.5 * (FindMaxAbs(dispYbottom) - FindMaxAbs(dispYtop));
                //std::cout << "dRx = " << dRx << ", dRy = " << dRy << "\n";

                //std::cout << (rad + dRx) * (rad + dRy) << "\n";
                HoleAreaPi += (rad + dRx) * (rad + dRy);
                if (k > 0 && l > 0 && k < nPores - 1 && l < nPores - 1) {
                    InternalHoleAreaPi += (rad + dRx) * (rad + dRy);
                }
            }
        }

        //std::cout << "HoleAreaPi = " << HoleAreaPi << "\n";
        const double Phi0 = 3.1415926 * rad * rad * nPores * nPores / (dX * (nX - 1) * dY * (nY - 1));
        /*std::cout << "Phi0 = " << Phi0 << '\n';
        log_file << "Phi0 = " << Phi0 << '\n';*/
        const double Phi = 3.1415926 * HoleAreaPi / (dX * (nX - 1) * dY * (nY - 1));
        const double PhiPer = nPores > 2 ? 
            3.1415926 * InternalHoleAreaPi / (dX * (nX - 1) * dY * (nY - 1) * (nPores - 2) * (nPores - 2) / nPores / nPores) :
            0.0;
        /*std::cout << "Phi = " << Phi << '\n';
        log_file << "Phi = " << Phi << '\n';*/
        dPhi[step][it] = 3.1415926 * std::abs(HoleAreaPi - rad * rad * nPores * nPores) / (dX * (nX - 1) * dY * (nY - 1));
        dPhiPer[step][it] = std::abs(PhiPer - Phi0);

        std::cout << "dPhi = " << dPhi[step][it] << '\n';
        log_file << "dPhi = " << dPhi[step][it] << '\n';
        if (nPores > 2) {
            std::cout << "dPhiPer = " << dPhiPer[step][it] << '\n';
            log_file << "dPhiPer = " << dPhiPer[step][it] << '\n';
        }
        //std::cout << "dPhi_new = " << GetdPhi() << "\n";

        /*const double Kphi = deltaP[step][it] / dPhi[step][it];
        const double KphiPer = deltaPper[step][it] / dPhiPer[step][it];*/

        //std::cout << "deltaP_honest = " << deltaP_honest << '\n';
        //log_file << "deltaP_honest = " << deltaP_honest << '\n';
        std::cout << "P / Y = " << deltaP[step][it] / Y << '\n';
        log_file << "P / Y = " << deltaP[step][it] / Y << '\n';
        std::cout << "Pper / Y = " << deltaPper[step][it] / Y << '\n';
        log_file << "Pper / Y = " << deltaPper[step][it] / Y << '\n';
        std::cout << "tau / Y = " << tauInfty[step][it] / Y << '\n';
        log_file << "tau / Y = " << tauInfty[step][it] / Y << '\n';
        /*std::cout << "KeffPhi = " << KeffPhi << '\n';
        log_file << "KeffPhi = " << KeffPhi << '\n';*/

        //const double phi = 3.1415926 * rad * rad / (dX * (nX - 1) * dY * (nY - 1));
        const double KexactElast = G0 / Phi0;
        const double KexactPlast = G0 / Phi / exp(std::abs(deltaP[step][it]) / Y - 1.0) / // phi or phi - dPhi ?
            (1.0 + tauInfty[step][it] * tauInfty[step][it] / Y / Y);
        const double KexactPlastPer = G0 / Phi0 / exp(std::abs(deltaPper[step][it]) / Y - 1.0) / // phi or phi - dPhi ?
            (1.0 + tauInfty[step][it] * tauInfty[step][it] / Y / Y);
        //std::cout << "KexactElast = " << KexactElast << '\n';
        log_file << "KexactElast = " << KexactElast << '\n';
        std::cout << "KexactPlast = " << KexactPlast << '\n';
        log_file << "KexactPlast = " << KexactPlast << '\n';
        std::cout << "KexactPlastPer = " << KexactPlastPer << '\n';
        log_file << "KexactPlastPer = " << KexactPlastPer << '\n';

        const double GexactElast = G0 / (1.0 + Phi0);
        const double GexactPlast = G0 / (1.0 + Phi * exp(std::abs(deltaP[step][it]) / Y - 1.0));
        const double GexactPlastPer = G0 / (1.0 + Phi0 * exp(std::abs(deltaPper[step][it]) / Y - 1.0));
        //std::cout << "GexactElast = " << GexactElast << '\n';
        log_file << "GexactElast = " << GexactElast << '\n';
        std::cout << "GexactPlast = " << GexactPlast << '\n';
        log_file << "GexactPlast = " << GexactPlast << '\n';
        std::cout << "GexactPlastPer = " << GexactPlastPer << '\n';
        log_file << "GexactPlastPer = " << GexactPlastPer << '\n';
    } // for(it), action loop
}

void EffPlast2D::ReadParams(const std::string& filename) {
    std::ifstream pa_fil(filename, std::ios_base::binary);
    if (!pa_fil) {
        std::cerr << "Error! Cannot open file " << filename << "!\n";
        exit(1);
    }
    pa_fil.read((char*)pa_cpu, sizeof(double) * NPARS);
    gpuErrchk(cudaMemcpy(pa_cuda, pa_cpu, NPARS * sizeof(double), cudaMemcpyHostToDevice));
}

void EffPlast2D::SetMaterials() {
    for (int i = 0; i < nX; i++) {
        for (int j = 0; j < nY; j++) {
            K_cpu[j * nX + i] = K0;
            G_cpu[j * nX + i] = G0;
            double x = -0.5 * dX * (nX - 1) + dX * i;
            double y = -0.5 * dY * (nY - 1) + dY * j;
            double Lx = dX * (nX - 1);
            double Ly = dY * (nY - 1);
            for (int k = 0; k < nPores; k++) {
                for (int l = 0; l < nPores; l++) {
                    if (sqrt((x - 0.5 * Lx * (1.0 - 1.0 / nPores) + (Lx / nPores) * k) * (x - 0.5 * Lx * (1.0 - 1.0 / nPores) + (Lx / nPores) * k) +
                        (y - 0.5 * Ly * (1.0 - 1.0 / nPores) + (Ly / nPores) * l) * (y - 0.5 * Ly * (1.0 - 1.0 / nPores) + (Ly / nPores) * l)) < rad) {
                        K_cpu[j * nX + i] = 0.01 * K0;
                        G_cpu[j * nX + i] = 0.01 * G0;
                        empty_spaces.emplace(i, j);
                    }
                }
            }
        }
    }
    /*for (int i = 0; i < nX; i++) {
        for (int j = 0; j < nY; j++) {
            if (empty_spaces.find({i, j}) == empty_spaces.end()) {
                if (empty_spaces.find({i + 1, j}) != empty_spaces.end() || empty_spaces.find({i - 1, j}) != empty_spaces.end()) {
                    empty_spaces.emplace(i, j);
                }
            }
        }
    }*/
    std::set<std::pair<int, int>> boundary_spaces;
    for (const auto& p : empty_spaces) {
        const int i = p.first;
        const int j = p.second;
        if (empty_spaces.find({i - 1, j}) == empty_spaces.end()) {
            boundary_spaces.emplace(i - 1, j);
        }
        if (empty_spaces.find({i + 1, j}) == empty_spaces.end()) {
            boundary_spaces.emplace(i + 1, j);
        }
    }
    for (auto i : boundary_spaces) {
        empty_spaces.insert(i);
    }
    boundary_spaces.clear();
    for (const auto& p : empty_spaces) {
        const int i = p.first;
        const int j = p.second;
        if (empty_spaces.find({i, j - 1}) == empty_spaces.end()) {
            boundary_spaces.emplace(i, j - 1);
        }
        if (empty_spaces.find({i, j + 1}) == empty_spaces.end()) {
            boundary_spaces.emplace(i, j + 1);
        }
    }
    for (auto i : boundary_spaces) {
        empty_spaces.insert(i);
    }

    gpuErrchk(cudaMemcpy(K_cuda, K_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(G_cuda, G_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice));
}

void EffPlast2D::SetInitPressure(const double coh) {
    const double P0 = 0.0; //1.0 * coh;

    for (int i = 0; i < nX; i++) {
        for (int j = 0; j < nY; j++) {
            P0_cpu[j * nX + i] = 0.0;
            if (sqrt((-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j)) < rad) {
                P0_cpu[j * nX + i] = P0;
            }
        }
    }

    gpuErrchk(cudaMemcpy(P0_cuda, P0_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice));
}

void EffPlast2D::SetMatrixZero(double** A_cpu, double** A_cuda, const int m, const int n) {
    *A_cpu = new double[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            (*A_cpu)[j * m + i] = 0.0;
        }
    }
    gpuErrchk(cudaMalloc(A_cuda, m * n * sizeof(double)));
    gpuErrchk(cudaMemcpy(*A_cuda, *A_cpu, m * n * sizeof(double), cudaMemcpyHostToDevice));
}

void EffPlast2D::SaveMatrix(double* const A_cpu, const double* const A_cuda, const int m, const int n, const std::string& filename) {
    gpuErrchk(cudaMemcpy(A_cpu, A_cuda, m * n * sizeof(double), cudaMemcpyDeviceToHost));
    std::ofstream A_filw(filename, std::ios_base::binary);
    A_filw.write((char*)A_cpu, sizeof(double) * m * n);
}

void EffPlast2D::SaveVector(double* const arr, const int size, const std::string& filename) {
    std::ofstream arr_filw(filename, std::ios_base::binary);
    arr_filw.write((char*)arr, sizeof(double) * size);
}

double EffPlast2D::FindMaxAbs(const double* const arr, const int size) {
    double max_el = 0.0;
    for (int i = 0; i < size; i++) {
        if (std::abs(arr[i]) > std::abs(max_el)) {
            max_el = std::abs(arr[i]);
        }
    }
    return max_el;
}

double EffPlast2D::FindMaxAbs(const std::vector<double>& vec) {
    double max_el = 0.0;
    for (auto i : vec) {
        if (std::abs(i) > std::abs(max_el)) {
            max_el = i;
        }
    }
    return max_el;
}

double EffPlast2D::GetDeltaP_honest() {
    double deltaP = 0.0, deltaPx = 0.0, deltaPy = 0.0;

    /*for (int i = 1; i < nX - 2; i++) {
        std::cout << "Sigma_xy = " << tauXY_cpu[0 * (nX - 1) + i] << "\n";
    }*/

    for (int i = 1; i < nX - 1; i++) {
        deltaPx += tauXX_cpu[0 * nX + i] - P_cpu[0 * nX + i];
        deltaPx += tauYY_cpu[0 * nX + i] - P_cpu[0 * nX + i];
        deltaPx += tauXX_cpu[(nY - 1) * nX + i] - P_cpu[(nY - 1) * nX + i];
        deltaPx += tauYY_cpu[(nY - 1) * nX + i] - P_cpu[(nY - 1) * nX + i];
        /*if ((i + 1) % 100 == 0) {
            std::cout << "P = " << 0.5 * (tauXX_cpu[0 * nX + i] - P_cpu[0 * nX + i] + tauYY_cpu[0 * nX + i] - P_cpu[0 * nX + i]) << "\n";
        }*/
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

double EffPlast2D::GetDeltaP_periodic() {
    if (nPores <= 2) {
        return 0.0;
    }
    double deltaP = 0.0, deltaPx = 0.0, deltaPy = 0.0;
    const int nP = static_cast<int>(nPores);

    for (int i = nX / nP; i < nX * (nP - 1) / nP; i++) {
        deltaPx += tauXX_cpu[nY / nP * nX + i] - P_cpu[nY / nP * nX + i];
        deltaPx += tauYY_cpu[nY / nP * nX + i] - P_cpu[nY / nP * nX + i];
        deltaPx += tauXX_cpu[nY * (nP - 1) / nP * nX + i] - P_cpu[nY * (nP - 1) / nP * nX + i];
        deltaPx += tauYY_cpu[nY * (nP - 1) / nP * nX + i] - P_cpu[nY * (nP - 1) / nP * nX + i];
        /*if ((i + 1) % 100 == 0) {
            std::cout << "P = " << 0.5 * (tauXX_cpu[0 * nX + i] - P_cpu[0 * nX + i] + tauYY_cpu[0 * nX + i] - P_cpu[0 * nX + i]) << "\n";
        }*/
    }
    deltaPx /= nX * (nP - 2) / nP;

    for (int j = nY / nP; j < nY * (nP - 1) / nPores; j++) {
        deltaPy += tauXX_cpu[j * nX + nX / nP] - P_cpu[j * nX + nX / nP];
        deltaPy += tauYY_cpu[j * nX + nX / nP] - P_cpu[j * nX + nX / nP];
        deltaPy += tauXX_cpu[j * nX + nX * (nP - 2) / nP] - P_cpu[j * nX + nX * (nP - 2) / nP];
        deltaPy += tauYY_cpu[j * nX + nX * (nP - 2) / nP] - P_cpu[j * nX + nX * (nP - 2) / nP];
    }
    deltaPy /= nY * (nP - 2) / nP;

    deltaP = -0.125 * (deltaPx + deltaPy);
    return deltaP;
}

double EffPlast2D::GetDeltaP_approx(const double Exx, const double Eyy) {
    double deltaP = 0.0;

    /*if (Exx < Eyy) {
        deltaP += tauXX_cpu[(nY / 2) * nX + 0] - P_cpu[(nY / 2) * nX + 0];
        deltaP += tauYY_cpu[(nY / 2) * nX + 0] - P_cpu[(nY / 2) * nX + 0];
        deltaP += tauXX_cpu[(nY / 2) * nX + nX - 1] - P_cpu[(nY / 2) * nX + nX - 1];
        deltaP += tauYY_cpu[(nY / 2) * nX + nX - 1] - P_cpu[(nY / 2) * nX + nX - 1];
    }
    else {
        deltaP += tauXX_cpu[0 * nX + nX / 2] - P_cpu[0 * nX + nX / 2];
        deltaP += tauYY_cpu[0 * nX + nX / 2] - P_cpu[0 * nX + nX / 2];
        deltaP += tauXX_cpu[(nY - 1) * nX + nX / 2] - P_cpu[(nY - 1) * nX + nX / 2];
        deltaP += tauYY_cpu[(nY - 1) * nX + nX / 2] - P_cpu[(nY - 1) * nX + nX / 2];
    }

    deltaP *= -0.25;*/

    deltaP += tauXX_cpu[(nY / 2) * nX + 0] - P_cpu[(nY / 2) * nX + 0];
    deltaP += tauYY_cpu[(nY / 2) * nX + 0] - P_cpu[(nY / 2) * nX + 0];
    deltaP += tauXX_cpu[(nY / 2) * nX + nX - 1] - P_cpu[(nY / 2) * nX + nX - 1];
    deltaP += tauYY_cpu[(nY / 2) * nX + nX - 1] - P_cpu[(nY / 2) * nX + nX - 1];
    deltaP += tauXX_cpu[0 * nX + nX / 2] - P_cpu[0 * nX + nX / 2];
    deltaP += tauYY_cpu[0 * nX + nX / 2] - P_cpu[0 * nX + nX / 2];
    deltaP += tauXX_cpu[(nY - 1) * nX + nX / 2] - P_cpu[(nY - 1) * nX + nX / 2];
    deltaP += tauYY_cpu[(nY - 1) * nX + nX / 2] - P_cpu[(nY - 1) * nX + nX / 2];

    deltaP *= -0.125;
    return deltaP;
}

double EffPlast2D::GetTauInfty_honestest() {
    double tauInfty = 0.0;
    for (int i = 1; i < nX - 1; i++) {
        for (int j = 1; j < nY - 1; j++) {
            double x = -0.5 * dX * (nX - 1) + dX * i;
            double y = -0.5 * dY * (nY - 1) + dY * j;
            if (sqrt(x * x + y * y) > rad) {
                tauInfty += sqrt(0.25 * (tauXX_cpu[j * nX + i] - tauYY_cpu[j * nX + i]) * (tauXX_cpu[j * nX + i] - tauYY_cpu[j * nX + i]) /*+
                    tauXY_cpu[j * (nX - 1) + i] * tauXY_cpu[j * (nX - 1) + i]*/);
            }
        }
    }
    tauInfty /= (nX - 2) * (nY - 2);
    return tauInfty;
}

double EffPlast2D::GetTauInfty_honest() {
    double tauInfty = 0.0, tauInftyx = 0.0, tauInftyy = 0.0;

    for (int i = 1; i < nX - 1; i++) {
        tauInftyx += tauXX_cpu[0 * nX + i] - tauYY_cpu[0 * nX + i];
        tauInftyx += tauXX_cpu[(nY - 1) * nX + i] - tauYY_cpu[(nY - 1) * nX + i];
    }
    tauInftyx /= (nX - 2);

    for (int j = 1; j < nY - 1; j++) {
        tauInftyy += tauXX_cpu[j * nX + 0] - tauYY_cpu[j * nX + 0];
        tauInftyy += tauXX_cpu[j * nX + nY - 1] - tauYY_cpu[j * nX + nY - 1];
    }
    tauInftyy /= (nY - 2);

    tauInfty = -0.125 * (tauInftyx + tauInftyy);
    return tauInfty;
}

double EffPlast2D::GetTauInfty_approx(const double Exx, const double Eyy) {
    double tauInfty = 0.0;

    /*if (Exx < Eyy) {
        tauInfty += tauYY_cpu[(nY / 2) * nX + 0] - tauXX_cpu[(nY / 2) * nX + 0];
        tauInfty += tauYY_cpu[(nY / 2) * nX + nX - 1] - tauXX_cpu[(nY / 2) * nX + nX - 1];
    }
    else {
        tauInfty += tauYY_cpu[0 * nX + nX / 2] - tauXX_cpu[0 * nX + nX / 2];
        tauInfty += tauYY_cpu[(nY - 1) * nX + nX / 2] - tauXX_cpu[(nY - 1) * nX + nX / 2];
    }

    tauInfty *= 0.25;*/

    /*std::cout << "Sigma_xy = " << tauXY_cpu[((nY - 1) / 2) * (nX - 1) + 0] << "\n";
    std::cout << "Sigma_xy = " << tauXY_cpu[((nY - 1) / 2) * (nX - 1) + nX - 2] << "\n";
    std::cout << "Sigma_xy = " << tauXY_cpu[0 * (nX - 1) + (nX  - 1) / 2] << "\n";
    std::cout << "Sigma_xy = " << tauXY_cpu[((nY - 2) / 2) * (nX - 2) + 0] << "\n";*/

    tauInfty += tauYY_cpu[(nY / 2) * nX + 0] - tauXX_cpu[(nY / 2) * nX + 0];
    tauInfty += tauYY_cpu[(nY / 2) * nX + nX - 1] - tauXX_cpu[(nY / 2) * nX + nX - 1];
    tauInfty += tauYY_cpu[0 * nX + nX / 2] - tauXX_cpu[0 * nX + nX / 2];
    tauInfty += tauYY_cpu[(nY - 1) * nX + nX / 2] - tauXX_cpu[(nY - 1) * nX + nX / 2];

    tauInfty *= 0.125;

    return tauInfty;
}

double EffPlast2D::GetdPhi() {
    const double phi0 = static_cast<double>(empty_spaces.size()) / ((nX - 1) * (nY - 1));
    double phi = 0.0;
    for (auto& p : empty_spaces) {
        const int i = p.first;
        const int j = p.second;
        //std::cout << "j = " << j << " i = " << i << "\n";
        //std::cout << "Ux = " << Ux_cpu[j * (nX + 1) + i + 1] << " and " << Ux_cpu[j * (nX + 1) + i] << "\n";
        //std::cout << "Uy = " << Uy_cpu[(j + 1) * nX + i] << " and " << Uy_cpu[j * nX + i] << "\n";
        phi += (1.0 + Ux_cpu[j * (nX + 1) + i + 1] - Ux_cpu[j * (nX + 1) + i]) * (1.0 + Uy_cpu[(j + 1) * nX + i] - Uy_cpu[j * nX + i]);
    }
    phi /= ((nX - 1) * (nY - 1));
    //std::cout << "Phi0new = " << phi0 << "\n";
    //std::cout << "Phi_new = " << phi << "\n";
    return phi - phi0;
}

void EffPlast2D::getAnalytic(
    double x, double y, double xi, double kappa, double c0,
    double& cosf, 
    double& sinf,
    std::complex<double>& zeta,
    std::complex<double>& w,
    std::complex<double>& dw,
    std::complex<double>& wv,
    std::complex<double>& Phi,
    std::complex<double>& Psi
)
{
    const std::complex<double> z = std::complex<double>(x, y);
    const double r = sqrt(x * x + y * y);
    cosf = x / r;
    sinf = y / r;

    double signx = x > 0.0 ? 1.0 : -1.0;
    if (abs(x) < std::numeric_limits<double>::epsilon())
        signx = 1.0;

    zeta = (z + signx * sqrt(z * z + 4.0 * c0 * c0 * kappa)) / 2.0 / c0;
    w    = c0 * (zeta - kappa / zeta);
    dw   = c0 * (1.0 + kappa / (zeta * zeta));
    wv   = c0 * (1.0 / zeta - kappa * zeta);
    Phi  = -Y * xi / 2.0 - Y * xi * log(w / zeta / rad);
    Psi  = -Y * xi / zeta * wv / dw;
}

std::complex<double> EffPlast2D::getAnalyticUelast(double x, double y, double tauInfty, double xi, double kappa, double c0)
{
    double cosf, sinf;
    std::complex<double> zeta, w, dw, wv, Phi, Psi;
    getAnalytic(x, y, xi, kappa, c0, cosf, sinf, zeta, w, dw, wv, Phi, Psi);

    const std::complex<double> phi  = -Y * xi * w * (log(w / zeta / rad) + 0.5) - 2.0 * c0 * tauInfty / zeta;
    const std::complex<double> psi  = c0 * Y * xi * (1.0 / zeta + kappa * zeta);
    const std::complex<double> dphi = Phi * dw;
    const std::complex<double> dpsi = Psi * dw;
    const std::complex<double> U    = ((1.0 + 6.0 * G0 / (G0 + 3.0 * K0)) * phi - w / conj(dw) * conj(dphi) - conj(psi)) / 2.0 / G0;

    return U;
}

double EffPlast2D::getAnalyticUrHydro(double r, double deltaP)
{
    return -0.5 * Y * rad * rad * exp((deltaP - Y) / Y) / (G0 * r);
}

double getJ1(double S11, double S22)
{
    return  0.5 * (S11 + S22);
}

double getJ2(double S11, double S22, double S12) {
    return 0.5 * ((S11 - S22) * (S11 - S22) + 4.0 * S12 * S12);
}

void cutError(double& e)
{
    if (std::abs(e) > 0.5)
        e = -0.5;
}

void EffPlast2D::getAnalyticJelast(double x, double y, double xi, double kappa, double c0, double& J1, double& J2)
{
    double cosf, sinf;
    std::complex<double> zeta, w, dw, wv, Phi, Psi;
    getAnalytic(x, y, xi, kappa, c0, cosf, sinf, zeta, w, dw, wv, Phi, Psi);

    const std::complex<double> z    = std::complex<double>(x, y);
    const std::complex<double> dPhi = -2.0 * xi * Y * kappa / zeta / (zeta * zeta - kappa);
    const std::complex<double> F    = 2.0 * (conj(w) / dw * dPhi + Psi) / exp(-2.0 * arg(z) * std::complex<double>(0.0, 1.0));

    const double Srr = 2.0 * real(Phi) - real(F) / 2.0;
    const double Sff = 2.0 * real(Phi) + real(F) / 2.0;
    const double Srf = imag(F) / 2.0;

    J1 = getJ1(Srr, Sff);
    J2 = getJ2(Srr, Sff, Srf);
}

void EffPlast2D::getAnalyticJplast(double r, double xi, double& J1, double& J2)
{
    const double Srr = -2.0 * xi * Y * log(r / rad);
    const double Sff = -2.0 * xi * Y * (1.0 + log(r / rad));
    const double Srf = 0.0;

    J1 = getJ1(Srr, Sff);
    J2 = getJ2(Srr, Sff, Srf);
}

void EffPlast2D::SaveAnStatic2D(const double deltaP, const double tauInfty, const std::array<double, 3>& loadType) {
    /* ANALYTIC 2D SOLUTION FOR STATICS */
    bool isHydro = 
        (abs(loadType[0] - loadType[1]) < std::numeric_limits<double>::min()) && 
        abs(loadType[2]) < std::numeric_limits<double>::min();

    const double Rmin = rad/* + 20.0 * dX*/;
    const double Rmax = 0.5 * dX * (nX - 1) - dX * 60.0;
    const double eps = 1.0e-18;

    const double xi = (deltaP > 0.0) ? 1.0 : -1.0;
    const double kappa = tauInfty / Y * xi;
    const double c0 = rad * exp(abs(deltaP) / 2.0 / Y - 0.5);

    const double rx = rad + getAnalyticUrHydro(rad, deltaP);
    const double ry = rx;

    const double Rx = c0 * (1.0 - kappa);
    const double Ry = c0 * (1.0 + kappa);

    double* UanAbs, * errorUabs, * J1an, * J2an, * errorJ1, * errorJ2; // , * plastZoneAn;
    if (nPores == 1)
    {
        UanAbs = new double[nX * nY];
        errorUabs = new double[nX * nY];
        J1an = new double[(nX - 1) * (nY - 1)];
        J2an = new double[(nX - 1) * (nY - 1)];
        errorJ1 = new double[(nX - 1) * (nY - 1)];
        errorJ2 = new double[(nX - 1) * (nY - 1)];
        //plastZoneAn = new double[(nX - 1) * (nY - 1)];
    }

    double errorUabsMax = 0.0, errorUabsAvg = 0.0;
    size_t errorUabsN = 0;

    double errorJ1Max = 0.0, errorJ1Avg = 0.0;
    double errorJ2Max = 0.0, errorJ2Avg = 0.0;
    size_t errorJN = 0;
    
    double* UnuAbs = new double[nX * nY];
    double* J1nu = new double[(nX - 1) * (nY - 1)];
    double* J2nu = new double[(nX - 1) * (nY - 1)];
    //double* plastZoneNu = new double[(nX - 1) * (nY - 1)];

    for (int i = 0; i < nX; i++) {
        for (int j = 0; j < nY; j++) {
            // numerical solution for Ur
            const double ux = 0.5 * (Ux_cpu[(nX + 1) * j + i] + Ux_cpu[(nX + 1) * j + (i + 1)]);
            const double uy = 0.5 * (Uy_cpu[nX * j + i] + Uy_cpu[nX * (j + 1) + i]);
            UnuAbs[j * nX + i] = sqrt(ux * ux + uy * uy);

            // numerical solution for sigma
            if (i < nX - 1 && j < nY - 1) {
                const double Sxx = 0.25 * (
                    -P_cpu[j * nX + i] + tauXX_cpu[j * nX + i] +
                    -P_cpu[j * nX + (i + 1)] + tauXX_cpu[j * nX + (i + 1)] +
                    -P_cpu[(j + 1) * nX + i] + tauXX_cpu[(j + 1) * nX + i] +
                    -P_cpu[(j + 1) * nX + (i + 1)] + tauXX_cpu[(j + 1) * nX + (i + 1)]
                );
                const double Syy = 0.25 * (
                    -P_cpu[j * nX + i] + tauYY_cpu[j * nX + i] +
                    -P_cpu[j * nX + (i + 1)] + tauYY_cpu[j * nX + (i + 1)] +
                    -P_cpu[(j + 1) * nX + i] + tauYY_cpu[(j + 1) * nX + i] +
                    -P_cpu[(j + 1) * nX + (i + 1)] + tauYY_cpu[(j + 1) * nX + (i + 1)]
                );
                const double Sxy = tauXY_cpu[j * (nX - 1) + i];

                J1nu[j * (nX - 1) + i] = getJ1(Sxx, Syy);
                J2nu[j * (nX - 1) + i] = getJ2(Sxx, Syy, Sxy);
            }

            // analytics
            if (nPores == 1) {
                // displacement, analytical solution for Ur
                const double x = -0.5 * dX * (nX - 1) + dX * i;
                const double y = -0.5 * dY * (nY - 1) + dY * j;

                const double r = sqrt(x * x + y * y);
                const double cosf = x / r;
                const double sinf = y / r;

                if (x * x / (Rx * Rx) + y * y / (Ry * Ry) > 1.0)
                {
                    // elast
                    if (isHydro)
                        UanAbs[j * nX + i] = abs(getAnalyticUrHydro(r, deltaP));
                    else
                        UanAbs[j * nX + i] = abs(getAnalyticUelast(x, y, tauInfty, xi, kappa, c0));
                }
                else if (x * x / (rx * rx) + y * y / (ry * ry) > 1.0)
                {
                    // plast
                    UanAbs[j * nX + i] = abs(getAnalyticUrHydro(r, deltaP));
                }
                else
                {
                    // hole
                    UanAbs[j * nX + i] = 0.0;
                }

                // relative error between analytical and numerical solution for Ur
                errorUabs[j * nX + i] = 0.0;
                if (
                    x * x + y * y > Rmin * Rmin &&
                    x * x + y * y < Rmax * Rmax &&
                    abs(UnuAbs[j * nX + i]) > eps
                    )
                {
                    errorUabs[j * nX + i] = abs((UanAbs[j * nX + i] - UnuAbs[j * nX + i]) / UanAbs[j * nX + i]);
                    errorUabsMax = std::max(errorUabsMax, errorUabs[j * nX + i]);
                    errorUabsAvg += errorUabs[j * nX + i];
                    errorUabsN++;

                    cutError(errorUabs[j * nX + i]);
                }

                // stress
                if (i < nX - 1 && j < nY - 1)
                {
                    const double x = -0.5 * dX * (nX - 1) + dX * i + 0.5 * dX;
                    const double y = -0.5 * dY * (nY - 1) + dY * j + 0.5 * dY;
                    const double r = sqrt(x * x + y * y);
                    const double cosf = x / r;
                    const double sinf = y / r;

                    // numerical plast zone
                    /*const double J2 = 0.25 * (J2_cpu[j * nX + i] + J2_cpu[j * nX + (i + 1)] + J2_cpu[(j + 1) * nX + i] + J2_cpu[(j + 1) * nX + (i + 1)]);

                    if (J2 > (1.0 - 2.0 * std::numeric_limits<double>::epsilon()) * pa_cpu[8])
                    {
                        plastZoneNu[j * (nX - 1) + i] = 1.0;
                    }
                    else
                    {
                        plastZoneNu[j * (nX - 1) + i] = 0.0;
                    }*/

                    // analytical solution for sigma
                    if (x * x / (Rx * Rx) + y * y / (Ry * Ry) > 1.0)
                    {
                        // elast
                        //plastZoneAn[j * (nX - 1) + i] = 0.0;
                        if (isHydro)
                        {
                            const double relR = rad / r;
                            const double Srr = -deltaP + relR * relR * Y * exp(deltaP / Y - 1);
                            const double Sff = -deltaP - relR * relR * Y * exp(deltaP / Y - 1);

                            J1an[j * (nX - 1) + i] = getJ1(Srr, Sff);
                            J2an[j * (nX - 1) + i] = getJ2(Srr, Sff, 0.0);
                        }
                        else
                            getAnalyticJelast(x, y, xi, kappa, c0, J1an[j * (nX - 1) + i], J2an[j * (nX - 1) + i]);
                    }
                    else if (x * x / (rx * rx) + y * y / (ry * ry) > 1.0)
                    {
                        // plast
                        //plastZoneAn[j * (nX - 1) + i] = 1.0;
                        getAnalyticJplast(r, xi, J1an[j * (nX - 1) + i], J2an[j * (nX - 1) + i]);
                    }
                    else
                    {
                        // hole
                        //plastZoneAn[j * (nX - 1) + i] = 0.0;
                        J1an[j * (nX - 1) + i] = 0.0;
                        J2an[j * (nX - 1) + i] = 0.0;
                    }

                    // relative error between analytical and numerical solution for sigma
                    errorJ1[j * (nX - 1) + i] = 0.0;
                    errorJ2[j * (nX - 1) + i] = 0.0;
                    if (
                        x * x + y * y > Rmin * Rmin &&
                        x * x + y * y < Rmax * Rmax
                        )
                    {
                        if (abs(J1nu[j * (nX - 1) + i]) > eps)
                        {
                            errorJ1[j * (nX - 1) + i] = abs((J1an[j * (nX - 1) + i] - J1nu[j * (nX - 1) + i]) / J1an[j * (nX - 1) + i]);
                            errorJ1Max = std::max(errorJ1Max, errorJ1[j * (nX - 1) + i]);
                            errorJ1Avg += errorJ1[j * (nX - 1) + i];

                            cutError(errorJ1[j * (nX - 1) + i]);
                        }

                        if (abs(J2nu[j * (nX - 1) + i]) > eps)
                        {
                            errorJ2[j * (nX - 1) + i] = abs((J2an[j * (nX - 1) + i] - J2nu[j * (nX - 1) + i]) / J2an[j * (nX - 1) + i]);
                            errorJ2Max = std::max(errorJ2Max, errorJ2[j * (nX - 1) + i]);
                            errorJ2Avg += errorJ2[j * (nX - 1) + i];

                            cutError(errorJ2[j * (nX - 1) + i]);
                        }

                        errorJN++;
                    }
                }
            }
        }
    }

    if (nPores == 1) {
        errorUabsAvg /= errorUabsN;
        errorJ1Avg /= errorJN;
        errorJ2Avg /= errorJN;

        std::cout << "\n"
            << "Uabs max error  = " << errorUabsMax << ", avg = " << errorUabsAvg << '\n'
            << "J1 max error = " << errorJ1Max << ", avg = " << errorJ1Avg << '\n'
            << "J2 max error = " << errorJ2Max << ", avg = " << errorJ2Avg << std::endl;

        log_file << "\n"
            << "Uabs max error  = " << errorUabsMax << ", avg = " << errorUabsAvg << '\n'
            << "J1 max error = " << errorJ1Max << ", avg = " << errorJ1Avg << '\n'
            << "J2 max error = " << errorJ2Max << ", avg = " << errorJ2Avg << std::endl;

        SaveVector(UanAbs, nX* nY, "data/UanAbs_" + std::to_string(32 * NGRID) + "_.dat");
        delete[] UanAbs;

        SaveVector(errorUabs, nX* nY, "data/errorUabs_" + std::to_string(32 * NGRID) + "_.dat");
        delete[] errorUabs;

        SaveVector(J1an, (nX - 1)* (nY - 1), "data/J1an_" + std::to_string(32 * NGRID) + "_.dat");
        delete[] J1an;

        SaveVector(J2an, (nX - 1)* (nY - 1), "data/J2an_" + std::to_string(32 * NGRID) + "_.dat");
        delete[] J2an;

        SaveVector(errorJ1, (nX - 1)* (nY - 1), "data/errorJ1_" + std::to_string(32 * NGRID) + "_.dat");
        delete[] errorJ1;

        SaveVector(errorJ2, (nX - 1)* (nY - 1), "data/errorJ2_" + std::to_string(32 * NGRID) + "_.dat");
        delete[] errorJ2;

        //SaveVector(plastZoneAn, (nX - 1)* (nY - 1), "data/plast_an_" + std::to_string(32 * NGRID) + "_.dat");
        //delete[] plastZoneAn;
    }
    
    SaveVector(UnuAbs, nX * nY, "data/UnuAbs_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] UnuAbs;

    SaveVector(J1nu, (nX - 1) * (nY - 1), "data/J1nu_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] J1nu;

    SaveVector(J2nu, (nX - 1) * (nY - 1), "data/J2nu_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] J2nu;

    //SaveVector(plastZoneNu, (nX - 1) * (nY - 1), "data/plast_nu_" + std::to_string(32 * NGRID) + "_.dat");
    //delete[] plastZoneNu;
}

EffPlast2D::EffPlast2D() {
    block.x = 32;
    block.y = 32;
    block.z = 1;
    grid.x = NGRID;
    grid.y = NGRID;
    grid.z = 1;

    nX = block.x * grid.x;
    nY = block.y * grid.y;

    gpuErrchk(cudaSetDevice(DEVICE_IDX));
    //gpuErrchk(cudaDeviceReset());
    //gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    /* PARAMETERS */
    pa_cpu = new double[NPARS];
    gpuErrchk(cudaMalloc(&pa_cuda, NPARS * sizeof(double)));
    ReadParams("data/pa.dat");

    dX = pa_cpu[0];
    dY = pa_cpu[1];
    dT = pa_cpu[2];
    K0 = pa_cpu[3];
    G0 = pa_cpu[4];
    E0 = 9.0 * K0 * G0 / (3.0 * K0 + G0);
    nu0 = (1.5 * K0 - G0) / (3.0 * K0 + G0);
    //std::cout << "E = " << E0 << ", nu = " << nu0 << "\n";
    rad = pa_cpu[9];
    Y = pa_cpu[8] / sqrt(2.0);
    nPores = pa_cpu[10];

    /* SPACE ARRAYS */
    // materials
    K_cpu = new double[nX * nY];
    G_cpu = new double[nX * nY];
    gpuErrchk(cudaMalloc(&K_cuda, nX * nY * sizeof(double)));
    gpuErrchk(cudaMalloc(&G_cuda, nX * nY * sizeof(double)));
    SetMaterials();

    // stress
    P0_cpu = new double[nX * nY];
    gpuErrchk(cudaMalloc(&P0_cuda, nX * nY * sizeof(double)));
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
    log_file.open("EffPlast2D.log", std::ios_base::app);
    output_step = 10'000;
    lX = (nX - 1) * dX;
    lY = (nY - 1) * dY;
    porosity = 3.1415926 * rad * rad * nPores * nPores / (lX * lY);
}

EffPlast2D::~EffPlast2D() {
    // parameters
    delete[] pa_cpu;
    gpuErrchk(cudaFree(pa_cuda));

    // materials
    delete[] K_cpu;
    delete[] G_cpu;
    gpuErrchk(cudaFree(K_cuda));
    gpuErrchk(cudaFree(G_cuda));

    // stress
    delete[] P0_cpu;
    delete[] P_cpu;
    delete[] tauXX_cpu;
    delete[] tauYY_cpu;
    delete[] tauXY_cpu;
    delete[] tauXYav_cpu;
    gpuErrchk(cudaFree(P0_cuda));
    gpuErrchk(cudaFree(P_cuda));
    gpuErrchk(cudaFree(tauXX_cuda));
    gpuErrchk(cudaFree(tauYY_cuda));
    gpuErrchk(cudaFree(tauXY_cuda));
    gpuErrchk(cudaFree(tauXYav_cuda));

    // plasticity
    delete[] J2_cpu;
    delete[] J2XY_cpu;
    gpuErrchk(cudaFree(J2_cuda));
    gpuErrchk(cudaFree(J2XY_cuda));

    // displacement
    delete[] Ux_cpu;
    delete[] Uy_cpu;
    gpuErrchk(cudaFree(Ux_cuda));
    gpuErrchk(cudaFree(Uy_cuda));

    // velocity
    delete[] Vx_cpu;
    delete[] Vy_cpu;
    gpuErrchk(cudaFree(Vx_cuda));
    gpuErrchk(cudaFree(Vy_cuda));

    // log
    log_file.close();
}