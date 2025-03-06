#include <string>
#include "EffPlast2D.h"

int main(int argc, char** argv) {
  try {
    const std::string err_info = "ERROR:  missing arguments\n USAGE: " + std::string(argv[0]) +
      " <init load value> <init load type>.x <init load type>.y <init load type>.xy <time steps> [<load value>]";
    if (argc < 6) {
      throw std::invalid_argument(err_info);
    }

    double init_load_value = std::stod(argv[1]);
    std::array<double, 3> load_type = {std::stod(argv[2]),  std::stod(argv[3]),  std::stod(argv[4])};
    unsigned int time_steps = std::stod(argv[5]);

    double load_value;
    if (time_steps == 1) {
      load_value = init_load_value;
    }
    else {
      if (argc < 7) {
        throw std::invalid_argument(err_info);
      }
      load_value = std::stod(argv[6]);
    }

    EffPlast2D eff_plast;
    eff_plast.ComputeEffModuli(init_load_value, load_value, time_steps, load_type);
    return 0;
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}