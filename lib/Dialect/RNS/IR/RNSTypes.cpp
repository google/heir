#include "include/Dialect/RNS/IR/RNSTypes.h"

#include "include/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace rns {

LogicalResult RNSType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<mlir::Type> basisTypes) {
  bool compatible = true;
  RNSBasisTypeInterface first =
      llvm::dyn_cast<RNSBasisTypeInterface>(basisTypes[0]);
  if (!first) return failure();

  for (auto other : basisTypes) {
    compatible &= first.isCompatibleWith(other);
  }

  if (!compatible) {
    return emitError() << "RNS type has incompatible basis types";
  }
  return success();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
