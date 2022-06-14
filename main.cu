#include <chrono>
#include "EffPlast2D.h"

int main() {
    std::vector<double> load_values = { -0.0004 };
    constexpr std::array<double, 3> load_type = { 1.0, 1.0, 0.0 };

    for (auto lv : load_values)
    {
        const auto start = std::chrono::system_clock::now();

        EffPlast2D eff_plast;
        const std::vector< std::array<double, 3> > S = eff_plast.ComputeSigma(lv, load_type);

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
    }

    return 0;
}