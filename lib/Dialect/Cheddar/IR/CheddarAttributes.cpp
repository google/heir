#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "llvm/include/llvm/ADT/ArrayRef.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

LogicalResult ParameterSetAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, int logN, int logScale,
    DenseI64ArrayAttr mainPrimes, DenseI64ArrayAttr auxPrimes,
    DenseI64ArrayAttr terminalPrimes, DenseI32ArrayAttr levelConfig,
    std::optional<int> wordBits, std::optional<int> defaultEncryptionLevel,
    DenseI32ArrayAttr additionalBase, std::optional<int> denseHammingWeight,
    std::optional<int> sparseHammingWeight, bool boot,
    std::optional<int> numCtsLevels, std::optional<int> numStcLevels) {
  if (logN <= 0) {
    return emitError() << "logN must be positive, got " << logN;
  }
  if (logScale <= 0) {
    return emitError() << "logScale must be positive, got " << logScale;
  }
  if (!mainPrimes || mainPrimes.empty()) {
    return emitError() << "mainPrimes must not be empty";
  }
  if (!auxPrimes || auxPrimes.empty()) {
    return emitError() << "auxPrimes must not be empty";
  }

  if (levelConfig) {
    ArrayRef<int32_t> levels = levelConfig.asArrayRef();
    if (levels.empty()) {
      return emitError() << "levelConfig must not be empty when provided";
    }
    if (levels.size() % 2 != 0) {
      return emitError() << "levelConfig size must be even, got "
                         << levels.size();
    }
    int64_t maxMain = mainPrimes.size();
    int64_t maxTerm = terminalPrimes ? terminalPrimes.size() : 0;
    for (size_t i = 0; i < levels.size(); i += 2) {
      int32_t numMain = levels[i];
      int32_t numTerm = levels[i + 1];
      if (numMain < 0 || numMain > maxMain) {
        return emitError() << "numMain at level " << (i / 2) << " (" << numMain
                           << ") must be in range [0, " << maxMain << "]";
      }
      if (numTerm < 0 || numTerm > maxTerm) {
        return emitError() << "numTerminal at level " << (i / 2) << " ("
                           << numTerm << ") must be in range [0, " << maxTerm
                           << "]";
      }
    }
    // levelConfig has one entry per level from 0 up to the max level, so the
    // default encryption level must index into it.
    int64_t numLevels = levels.size() / 2;
    if (defaultEncryptionLevel.has_value() &&
        (*defaultEncryptionLevel < 0 || *defaultEncryptionLevel >= numLevels)) {
      return emitError() << "defaultEncryptionLevel ("
                         << *defaultEncryptionLevel << ") must be in range [0, "
                         << numLevels << ") given levelConfig with "
                         << numLevels << " levels";
    }
  }

  if (additionalBase && additionalBase.size() != 2) {
    return emitError() << "additionalBase must have exactly 2 elements, got "
                       << additionalBase.size();
  }

  if (wordBits.has_value() && *wordBits != 32 && *wordBits != 64) {
    return emitError() << "wordBits must be 32 or 64, got " << *wordBits;
  }

  if (boot) {
    if (!numCtsLevels.has_value()) {
      return emitError() << "numCtsLevels is required when boot is true";
    }
    if (!numStcLevels.has_value()) {
      return emitError() << "numStcLevels is required when boot is true";
    }
  }

  return success();
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
