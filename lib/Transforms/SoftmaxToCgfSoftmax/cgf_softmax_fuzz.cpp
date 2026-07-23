// Fuzz test validating the approximation claims of CGF-softmax.
// Based on the paper: "CGF-Softmax: FHE-Friendly Softmax Approximation
// via Cumulant Generating Function" (https://arxiv.org/abs/2602.01621).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "absl/random/random.h"  // from @com_google_absl

// Exact softmax implementation
std::vector<double> exact_softmax(const std::vector<double>& x) {
  if (x.empty()) return {};
  double max_val = *std::max_element(x.begin(), x.end());
  std::vector<double> result(x.size());
  double sum = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = std::exp(x[i] - max_val);
    sum += result[i];
  }
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] /= sum;
  }
  return result;
}

// CGF softmax (second-order approximation) implementation
std::vector<double> cgf_softmax_2nd_order(const std::vector<double>& x) {
  if (x.empty()) return {};
  double n = static_cast<double>(x.size());

  // Compute mean (mu)
  double sum = std::accumulate(x.begin(), x.end(), 0.0);
  double mu = sum / n;

  // Compute variance (sigma^2)
  double sq_sum = 0.0;
  for (double val : x) {
    sq_sum += (val - mu) * (val - mu);
  }
  double sigma_sq = sq_sum / n;

  // Compute shift S = mu + sigma^2 / 2 + ln(n)
  double shift = mu + sigma_sq / 2.0 + std::log(n);

  std::vector<double> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = std::exp(x[i] - shift);
  }
  return result;
}

// Helper to compute max relative error
double compute_max_rel_error(const std::vector<double>& exact,
                             const std::vector<double>& approx) {
  double max_err = 0.0;
  for (size_t i = 0; i < exact.size(); ++i) {
    if (exact[i] == 0.0) {
      if (approx[i] != 0.0) return std::numeric_limits<double>::infinity();
      continue;
    }
    double err = std::abs(exact[i] - approx[i]) / exact[i];
    max_err = std::max(max_err, err);
  }
  return max_err;
}

// Run sweep to find max range width for target errors
void run_range_sweep() {
  absl::InsecureBitGen gen;
  std::vector<size_t> sizes = {8, 16, 32, 64, 128, 256};
  int num_trials = 100;  // Fewer trials for faster sweep

  std::cout << "Finding maximum range width W (input in [ -W/2, W/2 ]) for "
               "target errors:\n";
  std::cout << std::left << std::setw(10) << "Size" << std::setw(15)
            << "Distribution" << std::setw(20) << "W (Max Err < 1%)"
            << std::setw(20) << "W (Max Err < 5%)\n";
  std::cout << std::string(65, '-') << "\n";

  for (size_t size : sizes) {
    for (std::string dist_type : {"Uniform", "Normal"}) {
      double w_1pct = 0.0;
      double w_5pct = 0.0;

      // Sweep width from 0.1 to 10.0 with step 0.1
      for (double w = 0.1; w <= 10.0; w += 0.1) {
        double max_err_seen = 0.0;
        double half_w = w / 2.0;

        for (int t = 0; t < num_trials; ++t) {
          std::vector<double> x(size);
          if (dist_type == "Uniform") {
            for (size_t i = 0; i < size; ++i) {
              x[i] = absl::Uniform(gen, -half_w, half_w);
            }
          } else {
            for (size_t i = 0; i < size; ++i) {
              x[i] = absl::Gaussian(gen, 0.0, w / 6.0);
            }
          }

          auto exact = exact_softmax(x);
          auto approx = cgf_softmax_2nd_order(x);
          double err = compute_max_rel_error(exact, approx);
          max_err_seen = std::max(max_err_seen, err);
        }

        if (max_err_seen < 0.01) {
          w_1pct = w;
        }
        if (max_err_seen < 0.05) {
          w_5pct = w;
        }
      }

      std::cout << std::left << std::setw(10) << size << std::setw(15)
                << dist_type << std::setw(20) << w_1pct << std::setw(20)
                << w_5pct << "\n";
    }
  }
}

int main() {
  run_range_sweep();
  return 0;
}
