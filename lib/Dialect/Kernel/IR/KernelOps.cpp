#include "lib/Dialect/Kernel/IR/KernelOps.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/IR/OpImplementation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#define GET_OP_CLASSES
#include "lib/Dialect/Kernel/IR/KernelOps.cpp.inc"

namespace {
uint32_t ceil_log2(uint32_t x) {
  if (x <= 1) return 0;
  return std::bit_width(x - 1);
}

// Ported from OpenFHE's ComputeDegreesPS in
// third_party/openfhe/src/pke/lib/scheme/ckksrns/ckksrns-utils.cpp
std::pair<uint32_t, uint32_t> computeDegreesPS(uint32_t n) {
  if (n == 0) return {0, 0};

  constexpr uint32_t UPPER_BOUND_PS = 2204;
  static const std::vector<std::pair<uint32_t, uint32_t>> rangemap = {
      {2, 1},    {11, 2},   {13, 3},   {17, 2},   {55, 3},  {59, 4},
      {76, 3},   {239, 4},  {247, 5},  {284, 4},  {991, 5}, {1007, 6},
      {1083, 5}, {2015, 6}, {2031, 7}, {2204, 6},
  };

  if (n <= UPPER_BOUND_PS) {
    uint32_t m = 0;
    for (const auto& entry : rangemap) {
      if (n <= entry.first) {
        m = entry.second;
        break;
      }
    }
    uint32_t k = n / ((1U << m) - 1) + 1;
    return {k, m};
  }

  // Heuristic for larger degrees
  std::vector<uint32_t> klist;
  std::vector<uint32_t> mlist;
  std::vector<uint32_t> multlist;
  for (uint32_t k = 1; k <= n; ++k) {
    double log2_n_k = std::log2(static_cast<double>(n) / k);
    uint32_t max_m = static_cast<uint32_t>(std::ceil(log2_n_k + 1) + 1);
    for (uint32_t m = 1; m <= max_m; ++m) {
      if (n < (k * ((1U << m) - 1))) {
        double log2_k = std::log2(k);
        double log2_sqrt = std::log2(std::sqrt(static_cast<double>(n) / 2));
        if (std::abs(std::floor(log2_k) - std::floor(log2_sqrt)) <= 1.0) {
          klist.push_back(k);
          mlist.push_back(m);
          multlist.push_back(k + 2 * m + (1U << (m - 1)) - 4);
        }
      }
    }
  }
  if (multlist.empty()) {
    return {1, static_cast<uint32_t>(std::ceil(std::log2(n + 1)))};
  }
  uint32_t minIndex =
      std::min_element(multlist.begin(), multlist.end()) - multlist.begin();
  return {klist[minIndex], mlist[minIndex]};
}
}  // namespace

namespace mlir {
namespace heir {
namespace kernel {

int EvalChebyshevOp::getLevelsToDrop() {
  BackendName backend = BackendName::Lattigo;

  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) {
    emitWarning(
        "eval_chebyshev op could not find enclosing module for backend "
        "determination; using lattigo by default");
  } else {
    FailureOr<CompilationTarget> targetConfig = getTargetConfig(module);
    if (failed(targetConfig)) {
      emitWarning(
          "eval_chebyshev op could not determine chosen backend; using lattigo "
          "by default");
    } else {
      backend = targetConfig->backendName;
    }
  }

  auto coefficients = getCoefficients().getValue();
  if (coefficients.empty()) {
    // The zero polynomial consumes no depth
    return 0;
  }

  int baseDepth = 0;
  uint32_t degree = coefficients.size() - 1;
  switch (backend) {
    case BackendName::Lattigo:
      baseDepth = std::bit_width(static_cast<uint64_t>(degree));
      break;
    case BackendName::OpenFHE:
      if (degree == 0) {
        baseDepth = 0;
      } else if (degree < 5) {
        baseDepth = ceil_log2(degree) + 1;
      } else {
        auto [k, m] = computeDegreesPS(degree);
        baseDepth = ceil_log2(k) + m;
      }
      break;
    default:
      baseDepth = 0;
  }

  return baseDepth;
}

::mlir::OpOperand& EvalChebyshevOp::getOperandToReduce() {
  return getOperation()->getOpOperand(0);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
