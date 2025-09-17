#include "lib/Dialect/RNS/IR/RNSOps.h"

#include <cstdint>
#include <optional>

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
      cast<RNSType>(getElementTypeOrSelf(op.getInput().getType()));
  int64_t start = op.getStart().getZExtValue();
  int64_t size = op.getSize().getZExtValue();

  rns::RNSType truncatedType = rns::RNSType::get(
      context, elementType.getBasisTypes().drop_front(start).take_front(size));
  Type resultType = truncatedType;
  if (auto shapedType = dyn_cast<ShapedType>(op.getInput().getType())) {
    resultType = shapedType.clone(truncatedType);
  }

  results.push_back(resultType);
  return success();
}

LogicalResult ExtractSliceOp::verify() {
  auto rnsType = cast<RNSType>(getElementTypeOrSelf(getInput().getType()));
  int64_t numLimbs = rnsType.getBasisTypes().size();
  int64_t start = getStart().getZExtValue();
  int64_t size = getSize().getZExtValue();

  if (start < 0) {
    return emitOpError() << "start index " << start << " cannot be negative";
  }

  if (size < 0) {
    return emitOpError() << "size " << size << " cannot be negative";
  }

  if (start + size > numLimbs) {
    return emitOpError() << "slice of size " << size << " starting at " << start
                         << " is out of bounds for RNS type with " << numLimbs
                         << " limbs";
  }

  return success();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
