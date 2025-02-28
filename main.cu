#include <string>
#include "EffPlast2D.h"

int main(int argc, char** argv) {

  std::string err_info = "error: missing arguments\n usage: " + std::string(argv[0]) + 
    " <init load value> <init load type>.x <init load type>.y <init load type>.xy <time steps> [<load value>]";

  if (argc < 6) {
    std::cout << err_info;
    exit(-1);
  }

  double init_load_value = std::stod(argv[1]);
  std::array<double, 3> load_type = { std::stod(argv[2]),  std::stod(argv[3]),  std::stod(argv[4]) };
  unsigned int time_steps = std::stod(argv[5]);

  double load_value;
  if (time_steps == 1) {
    load_value = init_load_value;
  }
  else {
    if (argc < 7) {
      std::cout << err_info;
      exit(-1);
    }
    load_value = std::stod(argv[6]);
  }

  EffPlast2D eff_plast;
  /*auto S = */eff_plast.ComputeEffModuli(init_load_value, load_value, time_steps, load_type);

  return 0;
}