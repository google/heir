#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"

#include <cstddef>
#include <cstdint>

#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

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

  // Every prime must lie in (0, 2^(word-1)). Cheddar requires the spare top
  // bit, which is what lets the sum of two residues be computed in a single
  // word without wrapping.
  const uint64_t primeBound = uint64_t{1} << (word - 1);
  auto verifyPrimes = [&](::llvm::StringRef name,
                          ::llvm::ArrayRef<int64_t> primes) {
    for (int64_t prime : primes) {
      if (prime <= 0 || static_cast<uint64_t>(prime) >= primeBound) {
        emitError() << "'" << name << "' prime " << prime
                    << " is outside the range (0, 2^" << (word - 1) << ")";
        return false;
      }
    }
    return true;
  };
  if (!verifyPrimes("ter", ter.asArrayRef()) ||
      !verifyPrimes("main", main.asArrayRef()) ||
      !verifyPrimes("aux", aux.asArrayRef())) {
    return ::mlir::failure();
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
