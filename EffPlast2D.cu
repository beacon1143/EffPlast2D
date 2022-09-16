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

std::array<std::vector<std::array<double, 3>>, NL> EffPlast2D::ComputeSigma(
	const double initLoadValue, 
	const double loadValue, 
    const unsigned int nTimeSteps, 
	const std::array<double, 3>& loadType
)
{    
    log_file << "init load: (" << initLoadValue * loadType[0] << ", " << initLoadValue * loadType[1] << ", " << initLoadValue * loadType[2] << ")\n" 
             << "   + load: (" << loadValue * loadType[0] << ", " << loadValue * loadType[1] << ", " << loadValue * loadType[2] << ") x" << (nTimeSteps - 1) << '\n';
    std::cout << "init load: (" << initLoadValue * loadType[0] << ", " << initLoadValue * loadType[1] << ", " << initLoadValue * loadType[2] << ")\n" 
              << "   + load: (" << loadValue * loadType[0] << ", " << loadValue * loadType[1] << ", " << loadValue * loadType[2] << ") x" << (nTimeSteps - 1) << '\n';

    const double incPercent = 0.005;
    const double incLoad =  0.5 * (loadValue * loadType[0] + loadValue * loadType[1]) * incPercent;

    std::array<std::vector<std::array<double, 3>>, NL> Sigma;
    std::array<std::vector<double>, NL> deltaP;
    std::array<std::vector<double>, NL> dPhi;

    for (int nload = 0; nload < NL; nload++)
    {
        Sigma[nload].resize(nTimeSteps);
        deltaP[nload].resize(nTimeSteps);
        dPhi[nload].resize(nTimeSteps);

        double dUxdx = initLoadValue * loadType[0];
        double dUydy = initLoadValue * loadType[1];
        double dUxdy = initLoadValue * loadType[2];

        memset(Ux_cpu, 0, (nX + 1) * nY * sizeof(double));
        memset(Uy_cpu, 0, nX * (nY + 1) * sizeof(double));

        /* ACTION LOOP */
        for (int it = 0; it < nTimeSteps; it++) {
            log_file << "\n\nload step " << (it + 1) << '\n';
            std::cout << "\n\nload step " << (it + 1) << '\n';

            if (it > 0)
            {
                dUxdx = loadValue * loadType[0];
                dUydy = loadValue * loadType[1];
                dUxdy = loadValue * loadType[2];

                gpuErrchk(cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost));
            }

            for (int i = 0; i < nX + 1; i++) {
                for (int j = 0; j < nY; j++) {
                    Ux_cpu[j * (nX + 1) + i] += ((-0.5 * dX * nX + dX * i) * (dUxdx) + (-0.5 * dY * (nY - 1) + dY * j) * dUxdy) * (1.0 + nload * incPercent);
                }
            }
            gpuErrchk(cudaMemcpy(Ux_cuda, Ux_cpu, (nX + 1) * nY * sizeof(double), cudaMemcpyHostToDevice));

            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY + 1; j++) {
                    Uy_cpu[j * nX + i] += ((-0.5 * dY * nY + dY * j) * (dUydy)) * (1.0 + nload * incPercent);
                }
            }
            gpuErrchk(cudaMemcpy(Uy_cuda, Uy_cpu, nX * (nY + 1) * sizeof(double), cudaMemcpyHostToDevice));

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

                if ((iter + 1) % output_step == 0) {
                    gpuErrchk(cudaMemcpy(Vx_cpu, Vx_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost));
                    gpuErrchk(cudaMemcpy(Vy_cpu, Vy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost));

                    error = (FindMaxAbs(Vx_cpu, (nX + 1) * nY) / (dX * (nX - 1)) + FindMaxAbs(Vy_cpu, nX * (nY + 1)) / (dY * (nY - 1))) * dT /
                        (std::abs(loadValue) * std::max(std::max(std::abs(loadType[0]), std::abs(loadType[1])), std::abs(loadType[2])));

                    std::cout << "Iteration " << iter + 1 << ": Error is " << error << '\n';
                    log_file << "Iteration " << iter + 1 << ": Error is " << error << '\n';

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
            }
            /* AVERAGING */
            gpuErrchk(cudaMemcpy(P_cpu, P_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(tauXX_cpu, tauXX_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(tauYY_cpu, tauYY_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(tauXY_cpu, tauXY_cuda, (nX - 1) * (nY - 1) * sizeof(double), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(J2_cpu, J2_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost));

            /*Sigma[nload][it] = {0.0, 0.0, 0.0};
            for (int i = 0; i < nX; i++) {
              for (int j = 0; j < nY; j++) {
                Sigma[nload][it][0] += tauXX_cpu[j * nX + i] - P_cpu[j * nX + i];
                Sigma[nload][it][1] += tauYY_cpu[j * nX + i] - P_cpu[j * nX + i];
              }
            }
            Sigma[nload][it][0] /= nX * nY;
            Sigma[nload][it][1] /= nX * nY;

            for (int i = 0; i < nX - 1; i++) {
              for (int j = 0; j < nY - 1; j++) {
                Sigma[nload][it][2] += tauXY_cpu[j * (nX - 1) + i];
              }
            }
            Sigma[nload][it][2] /= (nX - 1) * (nY - 1);*/

            // -P_eff
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    if (sqrt((-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j)) >= rad) {
                        Sigma[nload][it][0] += -P_cpu[j * nX + i];
                    }
                    else {
                        // std::cout << "In the hole!\n";
                        // log_file << "In the hole!\n";
                    }
                }
            }
            Sigma[nload][it][0] /= nX * nY;

            // Tau_eff
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    if (sqrt((-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j)) >= rad) {
                        Sigma[nload][it][1] += tauXX_cpu[j * nX + i];
                        Sigma[nload][it][2] += tauYY_cpu[j * nX + i];
                    }
                }
            }
            Sigma[nload][it][1] /= nX * nY;
            Sigma[nload][it][2] /= nX * nY;

            // std::cout << Sigma[nload][it][0] / loadValue << '\t' << Sigma[nload][it][1] / loadValue << '\t' << Sigma[nload][it][2] / loadValue << '\n';
            // log_file << Sigma[nload][it][0] / loadValue << '\t' << Sigma[nload][it][1] / loadValue << '\t' << Sigma[nload][it][2] / loadValue << '\n';

            /* ANALYTIC SOLUTION FOR EFFECTIVE PROPERTIES */
            deltaP[nload][it] = GetDeltaP_approx(loadValue * loadType[0], loadValue * loadType[1]); // GetDeltaP_honest();
            std::cout << "deltaP = " << deltaP[nload][it] << '\n';
            log_file << "deltaP = " << deltaP[nload][it] << '\n';
            //const double deltaP = GetDeltaP_approx(loadValue * loadType[0], loadValue * loadType[1]);
            const double tauInfty_approx = GetTauInfty_approx(loadValue * loadType[0], loadValue * loadType[1]); // GetTauInfty_honest();

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

            /*std::vector<double> dispXwrong((nY + 1) / 2);
            for (int j = nY / 2 - holeY - 2; j < nY / 2; j++) {
              dispXwrong[j] = Ux_cpu[j * nX + nX / 2];
            }*/

            /*const double dR = FindMaxAbs(Ux_cpu, (nX + 1) * nY);
            std::cout << "dR = " << dR << '\n';
            log_file << "dR = " << dR << '\n';*/
            const double dRx = -FindMaxAbs(dispX);
            //std::cout << "dRx = " << dRx << '\n';
            log_file << "dRx = " << dRx << '\n';
            const double dRy = -FindMaxAbs(dispY);
            //std::cout << "dRy = " << dRy << '\n';
            log_file << "dRy = " << dRy << '\n';
            /*const double dRxWrong = -FindMaxAbs(dispXwrong);
            std::cout << "dRxWrong = " << dRxWrong << '\n';*/
            const double Phi0 = 3.1415926 * rad * rad / (dX * (nX - 1) * dY * (nY - 1));
            const double Phi = 3.1415926 * (rad + dRx) * (rad + dRy) / (dX * (nX - 1) * dY * (nY - 1) * (1 + loadValue * loadType[0]) * (1 + loadValue * loadType[1]));
            dPhi[nload][it] = 3.1415926 * (std::abs((rad + dRx) * (rad + dRy) - rad * rad)) / (dX * (nX - 1) * dY * (nY - 1));
            //std::cout << "dPhi = " << dPhi[nload][it] << '\n';
            log_file << "dPhi = " << dPhi[nload][it] << '\n';

            const double KeffPhi = deltaP[nload][it] / dPhi[nload][it];
            //const double KeffPhi = deltaP_honest / dPhi;

            //std::cout << "deltaP_honest = " << deltaP_honest << '\n';
            //log_file << "deltaP_honest = " << deltaP_honest << '\n';
            std::cout << "deltaP / Y = " << deltaP[nload][it] / Y << '\n';
            log_file << "deltaP / Y = " << deltaP[nload][it] / Y << '\n';
            //std::cout << "tauInfty / Y = " << tauInfty_approx / Y << '\n';
            log_file << "tauInfty / Y = " << tauInfty_approx / Y << '\n';
            //std::cout << "KeffPhi = " << KeffPhi << '\n';
            log_file << "KeffPhi = " << KeffPhi << '\n';

            const double phi = 3.1415926 * rad * rad / (dX * (nX - 1) * dY * (nY - 1));
            const double KexactElast = G0 / phi;
            const double KexactPlast = G0 / (phi - dPhi[nload][it]) / exp(std::abs(deltaP[nload][it]) / Y - 1.0) / // phi or phi - dPhi ?
                (1.0 + 5.0 * tauInfty_approx * tauInfty_approx / Y / Y);
            //const double KexactPlast = G0 / phi / exp(std::abs(deltaP_honest) / pa_cpu[8] - 1.0);
            //std::cout << "KexactElast = " << KexactElast << '\n';
            log_file << "KexactElast = " << KexactElast << '\n';
            std::cout << "KexactPlast = " << KexactPlast << '\n';
            log_file << "KexactPlast = " << KexactPlast << '\n';

            if (it + 1 == nTimeSteps && nload + 1 == NL)
                SaveAnStatic1D(deltaP[nload][it], tauInfty_approx);
        }
    }

    if (NL > 1)
    {
        const double KeffPhi = (deltaP[NL - 1][nTimeSteps - 1] - deltaP[NL - 2][nTimeSteps - 1]) / 
            (dPhi[NL - 1][nTimeSteps - 1] - dPhi[NL - 2][nTimeSteps - 1]);
        
        std::cout << "==============\n" << "KeffPhi = " << KeffPhi << '\n';
        log_file << "==============\n" << "KeffPhi = " << KeffPhi << '\n';
    }

    /* ANALYTIC 2D SOLUTION FOR STATICS */

    /* OUTPUT DATA WRITING */
    SaveMatrix(P_cpu, P_cuda, nX, nY, "Pc_" + std::to_string(32 * NGRID) + "_.dat");
    SaveMatrix(tauXX_cpu, tauXX_cuda, nX, nY, "tauXXc_" + std::to_string(32 * NGRID) + "_.dat");
    SaveMatrix(tauYY_cpu, tauYY_cuda, nX, nY, "tauYYc_" + std::to_string(32 * NGRID) + "_.dat");
    SaveMatrix(tauXY_cpu, tauXY_cuda, nX - 1, nY - 1, "tauXYc_" + std::to_string(32 * NGRID) + "_.dat");
    SaveMatrix(tauXYav_cpu, tauXYav_cuda, nX, nY, "tauXYavc_" + std::to_string(32 * NGRID) + "_.dat");
    SaveMatrix(J2_cpu, J2_cuda, nX, nY, "J2c_" + std::to_string(32 * NGRID) + "_.dat");
    SaveMatrix(Ux_cpu, Ux_cuda, nX + 1, nY, "Uxc_" + std::to_string(32 * NGRID) + "_.dat");
    SaveMatrix(Uy_cpu, Uy_cuda, nX, nY + 1, "Uyc_" + std::to_string(32 * NGRID) + "_.dat");

    //gpuErrchk(cudaDeviceReset());
    return Sigma;
}

void EffPlast2D::ReadParams(const std::string& filename) {
    std::ifstream pa_fil(filename, std::ios_base::binary);
    if (!pa_fil) {
        std::cerr << "Error! Cannot open file pa.dat!\n";
        exit(1);
    }
    pa_fil.read((char*)pa_cpu, sizeof(double) * NPARS);
    gpuErrchk(cudaMemcpy(pa_cuda, pa_cpu, NPARS * sizeof(double), cudaMemcpyHostToDevice));
}

void EffPlast2D::SetMaterials() {
    //constexpr double K0 = 10.0;
    //constexpr double G0 = 0.01;

    for (int i = 0; i < nX; i++) {
        for (int j = 0; j < nY; j++) {
            K_cpu[j * nX + i] = K0;
            G_cpu[j * nX + i] = G0;
            double x = -0.5 * dX * (nX - 1) + dX * i;
            double y = -0.5 * dY * (nY - 1) + dY * j;
            double Lx = dX * (nX - 1);
            double Ly = dY * (nY - 1);
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    if (sqrt((x - 0.5 * Lx * (1.0 - 1.0 / N) + (Lx / N) * k) * (x - 0.5 * Lx * (1.0 - 1.0 / N) + (Lx / N) * k) +
                        (y - 0.5 * Ly * (1.0 - 1.0 / N) + (Ly / N) * l) * (y - 0.5 * Ly * (1.0 - 1.0 / N) + (Ly / N) * l)) < rad) {
                        K_cpu[j * nX + i] = 0.01 * K0;
                        G_cpu[j * nX + i] = 0.01 * G0;
                    }
                }
            }
        }
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

    tauInfty += tauYY_cpu[(nY / 2) * nX + 0] - tauXX_cpu[(nY / 2) * nX + 0];
    tauInfty += tauYY_cpu[(nY / 2) * nX + nX - 1] - tauXX_cpu[(nY / 2) * nX + nX - 1];
    tauInfty += tauYY_cpu[0 * nX + nX / 2] - tauXX_cpu[0 * nX + nX / 2];
    tauInfty += tauYY_cpu[(nY - 1) * nX + nX / 2] - tauXX_cpu[(nY - 1) * nX + nX / 2];

    tauInfty *= 0.125;

    return tauInfty;
}

void EffPlast2D::SaveAnStatic1D(const double deltaP, const double tauInfty) {
    /* ANALYTIC 1D SOLUTION FOR STATICS */
    const double Rmin = rad + 0.0 * dX;
    const double Rmax = 0.5 * dX * (nX - 1) - dX * 60.0;

    const double xi = (deltaP > 0.0) ? 1.0 : -1.0;
    const double kappa = tauInfty / Y * xi;
    const double c0 = rad * exp(abs(deltaP) / 2.0 / Y - 0.5);

    const double Rx = c0 * (1.0 - kappa);
    const double Ry = c0 * (1.0 + kappa);

    // double Rxnu = 0.5 * (nX - 1) * dX;
    // for (int i = 0; i < nX / 2; i++)
    //     if (J2_cpu[(nY / 2) * nX + i] <= (1.0 - 2.0 * std::numeric_limits<double>::epsilon()) * pa_cpu[8])
    //         Rxnu -= dX;
    //     else
    //         break;

    // double Rynu = 0.5 * (nY - 1) * dY;
    // for (int i = 0; i < nY / 2; i++)
    //     if (J2_cpu[i * nX + (nX / 2)] <= (1.0 - 2.0 * std::numeric_limits<double>::epsilon()) * pa_cpu[8])
    //         Rynu -= dY;
    //     else
    //         break;

    // std::array<double, 2> conformParams = { 
    //     (Rxnu + Rynu) / 2.0,
    //     std::abs(Rxnu - Rynu) / 2.0
    // };
    // std::sort(conformParams.begin(), conformParams.end());

    // const double c0nu = conformParams[1];
    // const double kappaSignNu = (tauInfty * deltaP > 0.0) ? 1.0 : -1.0;
    // const double kappaNu = kappaSign * conformParams[0] / conformParams[1];

    // std::cout << "kappa num = " << kappaNu << "; kappa an = " << kappa << "\n"
    //           << "c0 num = " << c0nu << "; c0 an = " << c0 << "\n";

    double* Uanr = new double[nX * nY];
    double* Unur = new double[nX * nY];

    for (int i = 0; i < nX; i++)
    {
        for (int j = 0; j < nY; j++)
        {
            const double x = -0.5 * dX * (nX - 1) + dX * i;
            const double y = -0.5 * dY * (nY - 1) + dY * j;
            const double r = sqrt(x * x + y * y);
            const double cosf = x / r;
            const double sinf = y / r;

            if (
                x * x + y * y < Rmin * Rmin ||
                x * x + y * y > Rmax * Rmax
            )
            {
                Uanr[j * nX + i] = 0.0;
                Unur[j * nX + i] = 0.0;
            }
            else
            {
                if (x * x / (Rx * Rx) + y * y / (Ry * Ry) > 0) 
                {
                    const std::complex<double> z = std::complex<double>(x, y);

                    double signx = x > 0.0 ? 1.0 : -1.0;
                    if (abs(x) < std::numeric_limits<double>::epsilon())
                        signx = 1.0;

                    const std::complex<double> zeta = (z + signx * sqrt(z * z + 4.0 * c0 * c0 * kappa)) / 2.0 / c0;
                    const std::complex<double> w = c0 * (zeta - kappa / zeta);
                    const std::complex<double> dw = c0 * (1.0 + kappa / (zeta * zeta));
                    const std::complex<double> wv = c0 * (1.0 / zeta - kappa * zeta);
                    const std::complex<double> Phi = -Y * xi / 2.0 - Y * xi * log(w / zeta / rad); 
                    const std::complex<double> Psi = -Y * xi / zeta * wv / dw;
                    const std::complex<double> phi = - Y * xi * w * (log(w / zeta / rad) + 0.5) - 2.0 * c0 * tauInfty / zeta;
                    const std::complex<double> psi = c0 * Y * xi * (1.0 / zeta + kappa * zeta);
                    const std::complex<double> dphi = Phi * dw;
                    const std::complex<double> dpsi = Psi * dw;
                    const std::complex<double> U = (1.0 / (2.0 * G0) + 3.0 / (G0 + 3.0 * K0)) * phi - w / conj(dw) * conj(dphi) - conj(psi);

                    Uanr[j * nX + i] = real(U) * cosf + imag(U) * sinf;
                }
                else
                {
                    Uanr[j * nX + i] = -0.5 * Y * rad * rad * exp((deltaP - Y) / Y) / (G0 * r);
                }
                
                Unur[j * nX + i] = Ux_cpu[(nX + 1) * j + i] * cosf + Uy_cpu[nX * j + i] * sinf;
            }
        }
    }

    SaveVector(Uanr, nX * nY, "Uanr_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Uanr;

    SaveVector(Unur, nX * nY, "Unur_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Unur;

    double* Sanrr = new double[(nX - 1) * (nY - 1)];
    double* Sanff = new double[(nX - 1) * (nY - 1)];
    double* Sanrf = new double[(nX - 1) * (nY - 1)];

    double* Snurr = new double[(nX - 1) * (nY - 1)];
    double* Snuff = new double[(nX - 1) * (nY - 1)];
    double* Snurf = new double[(nX - 1) * (nY - 1)];

    double* plastZoneAn = new double[(nX - 1) * (nY - 1)];
    double* plastZoneNu = new double[(nX - 1) * (nY - 1)];

    for (int i = 0; i < nX - 1; i++)
    {
        for (int j = 0; j < nY - 1; j++)
        {
            const double x = -0.5 * dX * (nX - 1) + dX * i + 0.5 * dX;
            const double y = -0.5 * dY * (nY - 1) + dY * j + 0.5 * dY;
            const double r = sqrt(x * x + y * y);
            const double cosf = x / r;
            const double sinf = y / r;

            const std::complex<double> z = std::complex<double>(x, y);

            const double Rmin = rad + /*2*/0.0 * std::min(dX, dY);
            const double Rmax = 0.5 * dX * (nX - 1) - dX * 40;
            double signx = x > 0.0 ? 1.0 : -1.0;
            if (abs(x) < std::numeric_limits<double>::epsilon())
                signx = 1.0;

            const std::complex<double> zeta = (z + signx * sqrt(z * z + 4.0 * c0 * c0 * kappa)) / 2.0 / c0;
            const std::complex<double> w = c0 * (zeta - kappa / zeta);
            const std::complex<double> dw = c0 * (1.0 + kappa / (zeta * zeta));
            const std::complex<double> wv = c0 * (1.0 / zeta - kappa * zeta);
            const std::complex<double> phi = -Y * xi / 2.0 - Y * xi * log(w / zeta / rad); 
            const std::complex<double> psi = -Y * xi / zeta * wv / dw;
            const std::complex<double> dphi = - 2.0 * xi * Y * kappa / zeta / ( zeta * zeta - kappa );
            const std::complex<double> F = 2.0 * (conj(w) / dw * dphi + psi) / exp(-2.0 * arg(z) * std::complex<double>(0.0, 1.0));

            plastZoneAn[j * (nX - 1) + i] = 0.0;
            plastZoneNu[j * (nX - 1) + i] = 0.0;

            if (
                x * x + y * y < Rmin * Rmin ||
                x * x + y * y > Rmax * Rmax
            )
            {
                Sanrr[j * (nX - 1) + i] = 0.0;
                Sanff[j * (nX - 1) + i] = 0.0;
                Sanrf[j * (nX - 1) + i] = 0.0;

                Snurr[j * (nX - 1) + i] = 0.0;
                Snuff[j * (nX - 1) + i] = 0.0;
                Snurf[j * (nX - 1) + i] = 0.0;
            }
            else
            {
                const double relR = rad / r;
                const double J2 = 0.25 * (J2_cpu[j * nX + i] + J2_cpu[j * nX + (i + 1)] + J2_cpu[(j + 1) * nX + i] + J2_cpu[(j + 1) * nX + (i + 1)]);

                if (J2 > (1.0 - 2.0 * std::numeric_limits<double>::epsilon()) * pa_cpu[8]) 
                {
                    plastZoneNu[j * (nX - 1) + i] = 1.0;
                }

                if (x * x / (Rx * Rx) + y * y / (Ry * Ry) > 1.0) 
                {
                    // elast
                    // Sanrr[j * (nX - 1) + i] = -deltaP + relR * relR * Y * exp(deltaP / Y - 1);
                    // Sanff[j * (nX - 1) + i] = -deltaP - relR * relR * Y * exp(deltaP / Y - 1);

                    Sanrr[j * (nX - 1) + i] = 2.0 * real(phi) - real(F) / 2.0;
                    Sanff[j * (nX - 1) + i] = 2.0 * real(phi) + real(F) / 2.0;
                    Sanrf[j * (nX - 1) + i] = imag(F) / 2.0;
                }
                else 
                {
                    // plast
                    Sanrr[j * (nX - 1) + i] = -2.0 * Y * log(1.0 / relR);
                    Sanff[j * (nX - 1) + i] = -2.0 * Y * (1.0 + log(1.0 / relR));
                    Sanrf[j * (nX - 1) + i] = 0;

                    plastZoneAn[j * (nX - 1) + i] = 1.0;
                }

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

                Snurr[j * (nX - 1) + i] = Sxx * cosf * cosf + Syy * sinf * sinf + 2 * Sxy * sinf * cosf;
                Snuff[j * (nX - 1) + i] = Sxx * sinf * sinf + Syy * cosf * cosf - 2 * Sxy * sinf * cosf;
                Snurf[j * (nX - 1) + i] = (Syy - Sxx) * sinf * cosf + Sxy * (cosf * cosf - sinf * cosf);
            }
        }
    }

    SaveVector(Sanrr, (nX - 1) * (nY - 1), "Sanrr_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Sanrr;

    SaveVector(Sanff, (nX - 1) * (nY - 1), "Sanff_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Sanff;

    SaveVector(Sanrf, (nX - 1) * (nY - 1), "Sanrf_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Sanrf;

    SaveVector(Snurr, (nX - 1) * (nY - 1), "Snurr_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Snurr;

    SaveVector(Snuff, (nX - 1) * (nY - 1), "Snuff_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Snuff;

    SaveVector(Snurf, (nX - 1) * (nY - 1), "Snurf_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] Snurf;

    SaveVector(plastZoneAn, (nX - 1) * (nY - 1), "plast_an_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] plastZoneAn;

    SaveVector(plastZoneNu, (nX - 1) * (nY - 1), "plast_nu_" + std::to_string(32 * NGRID) + "_.dat");
    delete[] plastZoneNu;
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

    gpuErrchk(cudaSetDevice(0));
    //gpuErrchk(cudaDeviceReset());
    //gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    /* PARAMETERS */
    pa_cpu = new double[NPARS];
    gpuErrchk(cudaMalloc(&pa_cuda, NPARS * sizeof(double)));
    ReadParams("pa.dat");

    dX = pa_cpu[0];
    dY = pa_cpu[1];
    dT = pa_cpu[2];
    K0 = pa_cpu[3];
    G0 = pa_cpu[4];
    rad = pa_cpu[9];
    Y = pa_cpu[8] / sqrt(2.0);
    N = pa_cpu[10];

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