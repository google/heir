#ifndef LIB_TARGET_SIMFHE_SIMFHETEMPLATES_H_
#define LIB_TARGET_SIMFHE_SIMFHETEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace simfhe {

constexpr std::string_view kModulePrelude = R"""(
import params
import evaluator
from perf_counter import PerfCounter
from experiment import run_mutiple, print_table, Target
)""";

constexpr std::string_view kMainPrelude = R"""(
if __name__ == "__main__":
  targets = []

  # Note: Ideally, the parameters here would depend on the specific workload,
  # but currently, the emitter just uses the same parameters for all workloads.
  for scheme_params in [
      params.Alg_benchmark_baseline,
      params.Alg_benchmark_mod_down_merge,
      params.Alg_benchmark_mod_down_hoist,
      params.BEST_PARAMS,
  ]:
    print(scheme_params)
)""";

constexpr std::string_view kMainEpilogue = R"""(
# Run and print
headers, data = run_mutiple(targets)
print_table(headers, data)
print()
)""";

}  // namespace simfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_SIMFHE_SIMFHETEMPLATES_H_
