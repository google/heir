#include "lib/Dialect/RNS/IR/RNSOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
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
    DictionaryAttr attrs, mlir::PropertyRef properties,
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

// verification for ExtractSingleSlice used in both verify and inferReturnType
static LogicalResult verifyExtractSingleSliceInput(std::optional<Location> loc,
                                                   Type coeffType,
                                                   APInt index) {
  RNSType rnsCoeffType = dyn_cast<RNSType>(getElementTypeOrSelf(coeffType));
  if (!rnsCoeffType) return failure();
  int64_t sliceIndex = index.getSExtValue();

  int64_t numLimbs = rnsCoeffType.getBasisTypes().size();
  if (sliceIndex < 0 || sliceIndex >= numLimbs) {
    return emitOptionalError(
        loc, "'rns.extract_single_slice' index ", sliceIndex,
        " is out of bounds for an RNS type with ", numLimbs, " limbs");
  }

  auto limbCoeffType = dyn_cast<mod_arith::ModArithType>(
      rnsCoeffType.getBasisTypes()[sliceIndex]);
  if (!limbCoeffType) {
    return emitOptionalError(loc,
                             "'rns.extract_single_slice' requires the selected "
                             "RNS limb to have ModArith type, but found ",
                             rnsCoeffType.getBasisTypes()[sliceIndex]);
  }

  return success();
}

LogicalResult ExtractSingleSliceOp::verify() {
  return verifyExtractSingleSliceInput(getLoc(), getInput().getType(),
                                       getIndex());
}

LogicalResult ExtractSingleSliceOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::PropertyRef properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  ExtractSingleSliceOpAdaptor op(operands, attrs, properties, regions);
  Type ty = op.getInput().getType();
  APInt index = op.getIndex();
  if (failed(verifyExtractSingleSliceInput(loc, ty, index))) {
    return failure();
  }
  int64_t sliceIndex = index.getSExtValue();
  RNSType rnsCoeffType = cast<RNSType>(getElementTypeOrSelf(ty));
  auto truncatedType =
      cast<mod_arith::ModArithType>(rnsCoeffType.getBasisTypes()[sliceIndex]);

  Type resultType = truncatedType;
  if (auto shapedType = dyn_cast<ShapedType>(ty)) {
    resultType = shapedType.clone(truncatedType);
  }
  results.push_back(resultType);
  return success();
}

LogicalResult PackOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::PropertyRef properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  PackOpAdaptor op(operands, attrs, properties, regions);
  ValueRange input = op.getInput();
  // There must be at least one item in the list to form an RNS component
  if (input.empty()) {
    return emitOptionalError(loc, "'rns.pack' requires at least one input");
  }

  SmallVector<Type> basisTypes;
  basisTypes.reserve(input.size());
  for (Value operand : input) {
    auto maTy = dyn_cast<mod_arith::ModArithType>(operand.getType());
    if (!maTy) {
      return emitOptionalError(loc, "'rns.pack' got input with type ",
                               operand.getType());
    }
    basisTypes.push_back(maTy);
  }
  results.push_back(rns::RNSType::get(context, basisTypes));
  return success();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
