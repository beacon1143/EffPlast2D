#include <chrono>
#include "EffPlast2D.h"

int main(int argc, char** argv) {

    if (argc < 7)
    {
        std::cout << "error: missing arguments\n" 
            << "usage: " << argv[0] << " <init load value> <load value> <time steps> <load type>.x <load type>.y <load type>.z";
        exit(-1);
    }

    double init_load_value = std::stod(argv[1]);
    double load_value = std::stod(argv[2]);
    unsigned int time_steps = std::stod(argv[3]);
    std::array<double, 3> load_type = { std::stod(argv[4]),  std::stod(argv[5]),  std::stod(argv[6]) };

    const auto start = std::chrono::system_clock::now();

    EffPlast2D eff_plast;
    auto S = eff_plast.ComputeSigma(init_load_value, load_value, time_steps, load_type);

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