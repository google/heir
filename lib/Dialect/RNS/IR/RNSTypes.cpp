#include "lib/Dialect/RNS/IR/RNSTypes.h"

#include <cstddef>

#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "llvm/include/llvm/ADT/APInt.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

namespace rns {

LogicalResult RNSType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<mlir::Type> basisTypes) {
  bool compatible = true;

  auto getInterface = [&](Type type) -> FailureOr<RNSBasisTypeInterface> {
    auto res = mlir::dyn_cast<RNSBasisTypeInterface>(type);
    if (!res) {
      return emitError() << type << " does not have RNSBasisTypeInterface";
    }
    return res;
  };

  size_t numTypes = basisTypes.size();
  for (auto i = 0; i != numTypes; ++i) {
    for (auto j = i + 1; j != numTypes; ++j) {
      auto resI = getInterface(basisTypes[i]);
      if (failed(resI)) {
        return resI;
      }
      auto resJ = getInterface(basisTypes[j]);
      if (failed(resJ)) {
        return resJ;
      }
      compatible &= (*resI).isCompatibleWith(*resJ);
    }
  }

  if (!compatible) {
    return emitError() << "RNS type has incompatible basis types";
  }
  return success();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
