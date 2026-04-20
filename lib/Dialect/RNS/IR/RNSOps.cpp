#include "lib/Dialect/RNS/IR/RNSOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace rns {

FailureOr<SmallVector<Value>> computeMixedRadixCoeffs(
    ImplicitLocOpBuilder& b, Value input, const ArrayAttr& qInvProds) {
  auto rnsTy = dyn_cast<RNSType>(input.getType());
  if (!rnsTy) {
    emitError(b.getLoc()) << "expected an RNS value, got " << input.getType();
    return failure();
  }
  int64_t numLimbs = rnsTy.getBasisTypes().size();
  // qInvProds is the partial products of all moduli: q0, q0*q1, q0*q1*q2, ...,
  // excluding the product of all the moduli, hence numLimbs-1.
  if (qInvProds.size() != numLimbs - 1) {
    emitError(b.getLoc()) << "expected " << (numLimbs - 1)
                          << " qInvProds for an RNS basis with " << numLimbs
                          << " limbs, got " << qInvProds.size();
    return failure();
  }

  ArrayRef<Attribute> qInvAttrs = qInvProds.getValue();
  SmallVector<Value> mrcs;

  // The first mixed-radix coefficient just the lift of the first limb to the
  // integers
  auto c0Reduced = ExtractSingleSliceOp::create(b, input, b.getIndexAttr(0));
  Type intType =
      cast<mod_arith::ModArithType>(c0Reduced.getType()).getLoweringType();
  Value c0Lifted = mod_arith::ExtractOp::create(b, intType, c0Reduced);
  mrcs.push_back(c0Lifted);

  // Subsequent coefficients depend on prior coefficients
  // c_i = \parens*{x_i - \sum_{j=0}^{i-1} \bracks*{c_j}_{q_j}\cdot
  // Q_{j-1}}\cdot Q_{i-1}^{-1} \in\Z_{q_i} Here the Q_i's represent partial
  // products of the limb moduli. Rather than use these values directly, we
  // evaluate the sum using Horner's method.
  for (int i = 1; i < numLimbs; i++) {
    Value xi = ExtractSingleSliceOp::create(b, input, b.getIndexAttr(i));
    auto limbTy = dyn_cast<mod_arith::ModArithType>(xi.getType());
    if (!limbTy) {
      emitError(b.getLoc())
          << "expected limb type to be mod_arith attribute, got "
          << xi.getType();
      return failure();
    }
    // Using Horner's method, we compute (c_{i-1}*q_{i-2} + c_{i-2})*q_{i-3} +
    // c_{i-3} ... We start by reducing c_{i-1} (the previous mixed-radix coeff)
    // mod, and accumulate into temp.
    Value temp = mod_arith::EncapsulateOp::create(b, limbTy, mrcs[i - 1]);
    for (int j = i - 2; j >= 0; j--) {
      // reduce c_j mod q_i
      Value reducedCj = mod_arith::EncapsulateOp::create(b, limbTy, mrcs[j]);
      Value qjConst =
          mod_arith::ConstantOp::create(b, limbTy, limbTy.getModulus());
      temp = mod_arith::MacOp::create(b, temp, qjConst, reducedCj);
    }
    Value ci = mod_arith::SubOp::create(b, xi, temp);
    auto maAttr = dyn_cast<mod_arith::ModArithAttr>(qInvAttrs[i - 1]);
    if (!maAttr) {
      emitError(b.getLoc())
          << "expected mod_arith attribute, got " << qInvAttrs[i - 1];
      return failure();
    }
    if (maAttr.getType() != limbTy) {
      emitError(b.getLoc())
          << "expected qInv attribute type to match limb type " << limbTy
          << ", got " << maAttr.getType();
      return failure();
    }
    Value qInvConst =
        mod_arith::ConstantOp::create(b, limbTy, maAttr.getValue());
    ci = mod_arith::MulOp::create(b, ci, qInvConst);
    Value liftedCi = mod_arith::ExtractOp::create(b, intType, ci);
    mrcs.push_back(liftedCi);
  }
  return mrcs;
}

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
