#include <chrono>
#include "EffPlast2D.h"

int main(int argc, char** argv) {

    if (argc < 5)
    {
        std::cout << "error: missing arguments\n";
        exit(-1);
    }

    double load_value = std::stod(argv[1]);
    std::array<double, 3> load_type = { std::stod(argv[2]),  std::stod(argv[3]),  std::stod(argv[4]) };

    const auto start = std::chrono::system_clock::now();

    EffPlast2D eff_plast;
    const std::vector< std::array<double, 3> > S = eff_plast.ComputeSigma(load_value, load_type);

    const auto end = std::chrono::system_clock::now();

    const int elapsed_sec = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(end - start).count());
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