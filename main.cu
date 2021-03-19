#include <chrono>
#include "EffPlast2D.h"

int main() {
  const auto start = std::chrono::system_clock::now();

  constexpr double load_value = -0.002;
  constexpr std::array<double, 3> load_type = {1.0, 1.0, 0.0};
  /*const std::vector< std::array<double, 3> > Sxx = ComputeSigma(load_value, {1, 0, 0});
  const std::vector< std::array<double, 3> > Syy = ComputeSigma(load_value, {0, 1, 0});
  const std::vector< std::array<double, 3> > Sxy = ComputeSigma(load_value, {0, 0, 1});

  std::vector<double> C_1111(NT), C_1122(NT), C_1112(NT),
                      C_2222(NT), C_1222(NT),
                      C_1212(NT);
  
  for (int it = 0; it < NT; it++) {
    C_1111[it] = Sxx[it][0] / load_value / (it + 1) * NT;
    C_1122[it] = Sxx[it][1] / load_value / (it + 1) * NT;
    C_1112[it] = Sxx[it][2] / load_value / (it + 1) * NT;

    C_2222[it] = Syy[it][1] / load_value / (it + 1) * NT;
    C_1222[it] = Syy[it][2] / load_value / (it + 1) * NT;

    C_1212[it] = Sxy[it][2] / load_value / (it + 1) * NT;

    std::cout << "C_1111[" << it << "] = " << C_1111[it] << '\n';
    std::cout << "C_1122[" << it << "] = " << C_1122[it] << '\n';
    std::cout << "C_1112[" << it << "] = " << C_1112[it] << '\n';
    std::cout << "C_2222[" << it << "] = " << C_2222[it] << '\n';
    std::cout << "C_1222[" << it << "] = " << C_1222[it] << '\n';
    std::cout << "C_1212[" << it << "] = " << C_1212[it] << '\n';
  }*/

  EffPlast2D eff_plast;
  const std::vector< std::array<double, 3> > S = eff_plast.ComputeSigma(load_value, load_type);

  const double divUeff = load_value * (load_type[0] + load_type[1]);
  std::vector<double> Keff(NT);
  std::vector<std::array<double, 2>> Geff(NT);
  for (int it = 0; it < NT; it++) {
    Keff[it] = S[it][0] / divUeff / (it + 1) * NT;
    Geff[it][0] = 0.5 * S[it][1] / (load_value * load_type[0] - divUeff / 3.0) / (it + 1) * NT;
    Geff[it][1] = 0.5 * S[it][2] / (load_value * load_type[1] - divUeff / 3.0) / (it + 1) * NT;

    std::cout << "K_eff[" << it << "] = " << Keff[it] << '\n';
    std::cout << "Gxx_eff[" << it << "] = " << Geff[it][0] << '\n';
    std::cout << "Gyy_eff[" << it << "] = " << Geff[it][1] << '\n';
  }

  const auto end = std::chrono::system_clock::now();
  const int elapsed_sec = static_cast<int>( std::chrono::duration_cast<std::chrono::seconds>(end - start).count() );
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