#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"

#include <cstddef>
#include <cstdint>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

::mlir::LogicalResult ParamsAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, uint32_t word,
    uint32_t log_n, ::mlir::DenseI64ArrayAttr ter,
    ::mlir::DenseI64ArrayAttr main, ::mlir::DenseI64ArrayAttr aux,
    ::mlir::ArrayAttr level_config, uint32_t alpha, uint32_t dnum) {
  if (word != 32 && word != 64) {
    return emitError() << "word must be 32 or 64, but got " << word;
  }
  if (log_n == 0) {
    return emitError() << "log_n must be positive, but got " << log_n;
  }
  if (alpha == 0) {
    return emitError() << "alpha must be positive, but got " << alpha;
  }
  return ::mlir::success();
}

size_t ParamsAttr::getNumLimbs() const {
  return getTer().size() + getMain().size() + getAux().size();
}

uint64_t ParamsAttr::getPrimeAt(size_t limbIdx) const {
  size_t numTerPrimes = getTer().size();
  size_t numMainPrimes = getMain().size();
  size_t numAuxPrimes = getAux().size();

  if (limbIdx < numTerPrimes) {
    // Terminal primes in reversed order: ter[num_ter - 1 - limbIdx]
    return static_cast<uint64_t>(getTer()[numTerPrimes - 1 - limbIdx]);
  }
  limbIdx -= numTerPrimes;
  if (limbIdx < numMainPrimes) {
    return static_cast<uint64_t>(getMain()[limbIdx]);
  }
  limbIdx -= numMainPrimes;
  if (limbIdx < numAuxPrimes) {
    return static_cast<uint64_t>(getAux()[limbIdx]);
  }
  return 0;
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
