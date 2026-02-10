#include "lib/Dialect/RNS/IR/RNSAttributes.h"

#include "mlir/include/mlir/IR/Attributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace rns {

LogicalResult RNSAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<mlir::IntegerAttr> values,
    ::mlir::heir::rns::RNSType type) {
  auto basisSize = type.getBasisTypes().size();
  if (values.size() != basisSize) {
    return emitError() << "expected " << basisSize
                       << " values to match the RNS basis size, but found "
                       << values.size();
  }
  return success();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
