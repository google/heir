#include "lib/Dialect/RNS/IR/RNSOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace rns {

LogicalResult ExtractSliceOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  ExtractSliceOpAdaptor op(operands, attrs, properties, regions);
  RNSType elementType =
      dyn_cast<RNSType>(getElementTypeOrSelf(op.getInput().getType()));
  if (!elementType) return failure();
  RNSType truncatedType =
      inferExtractSliceReturnTypes(context, &op, elementType);
  Type resultType = truncatedType;
  if (auto shapedType = dyn_cast<ShapedType>(op.getInput().getType())) {
    resultType = shapedType.clone(truncatedType);
  }
  results.push_back(resultType);
  return success();
}

LogicalResult ExtractSliceOp::verify() {
  auto rnsType = dyn_cast<RNSType>(getElementTypeOrSelf(getInput().getType()));
  if (!rnsType) {
    return failure();
  }
  int64_t start = getStart().getZExtValue();
  int64_t size = getSize().getZExtValue();

  return verifyExtractSliceOp(this, rnsType, start, size);
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
