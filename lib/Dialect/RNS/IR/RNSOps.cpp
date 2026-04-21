#include "lib/Dialect/RNS/IR/RNSOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallString.h"           // from @llvm-project
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

FailureOr<ArrayAttr> buildQInvProds(mlir::MLIRContext* ctx,
                                    rns::RNSType basisTy) {
  SmallVector<Attribute> qInvProdAttrs;
  ArrayRef<Type> basisTypes = basisTy.getBasisTypes();
  if (basisTypes.empty()) {
    return ArrayAttr::get(ctx, qInvProdAttrs);
  }
  qInvProdAttrs.reserve(basisTypes.size() - 1);

  SmallVector<mod_arith::ModArithType> maBases;
  for (size_t i = 0; i < basisTypes.size(); ++i) {
    auto qiTy = dyn_cast<mod_arith::ModArithType>(basisTypes[i]);
    if (!qiTy) {
      return failure();
    }
    maBases.push_back(qiTy);
  }

  // If there are k basis elements, we compute
  // (prod_{j=0..i} q_i)^{-1} mod q_{i+1} for i from 0 to k-2
  APInt partialProduct(/*numBits=*/1, /*val=*/1);
  for (size_t i = 0; i < basisTypes.size() - 1; ++i) {
    mod_arith::ModArithType qiTy = maBases[i];
    APInt qi = qiTy.getModulus().getValue();
    unsigned productWidth = partialProduct.getBitWidth() + qi.getBitWidth();
    partialProduct = partialProduct.zext(productWidth) * qi.zext(productWidth);

    auto modTy = maBases[i + 1];
    APInt modValue = modTy.getModulus().getValue();

    // Reduce the partial product mod modValue before computing the inverse
    // to reduce bitwidth
    APInt reducedPartial =
        partialProduct.urem(modValue.zext(partialProduct.getBitWidth()))
            .trunc(modValue.getBitWidth());
    APInt qInv = multiplicativeInverse(reducedPartial, modValue);
    if (qInv.isZero()) {
      return failure();
    }

    IntegerAttr qInvValue = IntegerAttr::get(
        modTy.getModulus().getType(),
        qInv.zextOrTrunc(modTy.getModulus().getValue().getBitWidth()));
    qInvProdAttrs.push_back(
        mod_arith::ModArithAttr::get(ctx, modTy, qInvValue));
  }

  return ArrayAttr::get(ctx, qInvProdAttrs);
}

// https://userpages.cs.umbc.edu/lomonaco/s08/441/handouts/GarnerAlg.pdf
//
// NOTE: When lifting using standard representatives, Garner's algorithm (not
// mod p) outputs the standard representative of the CRT reconstruction without
// any explicit mods by the product of the input basis. When lifting using
// canonical representatives, it outputs the canonical representative of the CRT
// reconstruction *WHEN THE INPUT BASIS IS ODD*. The key here is that when the
// cs are (perfectly) centered, the reconstruction will be as well. If some
// modulus is even, however, the canonical representative for that component is
// *not* perfectly centered. As a result, the reconstructed output is also not
// perfectly centered, i.e., it is not the canonical representative! This breaks
// everything because we rely on the fact that Garner's outputs lift(crt(xs,
// qs)) so that when we compute it mod p, we actually are doing basis extension.
// If Garner outputs y != lift(crt(xs, qs)), then we compute y mod p, which is
// meaningless.
FailureOr<Value> convertBasis(
    ImplicitLocOpBuilder b, ArrayAttr qInvProds, Value x,
    rns::RNSType targetBasisTy,
    const llvm::DenseMap<APInt, size_t>& inputBasisIndexByModulus) {
  rns::RNSType inputBasisTy = dyn_cast<rns::RNSType>(x.getType());
  if (!inputBasisTy) {
    emitError(b.getLoc()) << "expected RNS coefficient, got " << x.getType();
    return failure();
  }

  ArrayRef<Type> inputBasisTypes = inputBasisTy.getBasisTypes();
  ArrayRef<Type> targetBasisTypes = targetBasisTy.getBasisTypes();
  FailureOr<SmallVector<Value>> maybeMrcs =
      rns::computeMixedRadixCoeffs(b, x, qInvProds);
  if (failed(maybeMrcs)) {
    return failure();
  }
  if (maybeMrcs->empty()) {
    emitError(b.getLoc()) << "expected non-empty mixed-radix coefficients";
    return failure();
  }
  SmallVector<Value>& mrcs = *maybeMrcs;

  SmallVector<mod_arith::ModArithType> maInputBasisTys;
  for (int i = 0; i < inputBasisTypes.size(); i++) {
    mod_arith::ModArithType qi =
        dyn_cast<mod_arith::ModArithType>(inputBasisTypes[i]);
    if (!qi) {
      emitError(b.getLoc()) << "expected source basis limb to be ModArithType";
      return failure();
    }
    APInt modulusValue = qi.getModulus().getValue();
    if (!modulusValue[0]) {
      SmallString<16> modulusStr;
      modulusValue.toStringUnsigned(modulusStr);
      emitError(b.getLoc())
          << "basis conversion requires odd moduli, but input basis contains "
             "even modulus "
          << modulusStr;
      return failure();
    }
    maInputBasisTys.push_back(qi);
  }
  IntegerType storageTy =
      cast<IntegerType>(cast<mod_arith::ModArithType>(targetBasisTypes[0])
                            .getModulus()
                            .getType());

  SmallVector<Value> outputLimbs;
  outputLimbs.reserve(targetBasisTypes.size());
  for (int i = 0; i < targetBasisTypes.size(); i++) {
    mod_arith::ModArithType targetLimbTy =
        dyn_cast<mod_arith::ModArithType>(targetBasisTypes[i]);
    if (!targetLimbTy) {
      emitError(b.getLoc()) << "expected target basis limb to be ModArithType";
      return failure();
    }
    APInt targetModulusValue = targetLimbTy.getModulus().getValue();
    if (!targetModulusValue[0]) {
      SmallString<16> modulusStr;
      targetModulusValue.toStringUnsigned(modulusStr);
      emitError(b.getLoc())
          << "basis conversion requires odd moduli, but target basis contains "
             "even modulus "
          << modulusStr;
      return failure();
    }

    llvm::DenseMap<APInt, size_t>::const_iterator inputIndexIt =
        inputBasisIndexByModulus.find(targetModulusValue);
    if (inputIndexIt != inputBasisIndexByModulus.end()) {
      outputLimbs.push_back(rns::ExtractSingleSliceOp::create(
          b, x, b.getIndexAttr(inputIndexIt->second)));
      continue;
    }

    // If the output modulus isn't in the input basis, compute its
    // representative Again, this uses Horner's method with `temp` as the
    // accumulator
    Value temp = mod_arith::EncapsulateOp::create(b, targetLimbTy, mrcs.back());
    for (int j = static_cast<int>(mrcs.size()) - 2; j >= 0; j--) {
      // get q_j, extend it to q_i's width, and reduce it mod q_i.
      mod_arith::ModArithType sourceLimbTy = maInputBasisTys[j];
      IntegerAttr qjAttr = IntegerAttr::get(
          storageTy, sourceLimbTy.getModulus().getValue().zextOrTrunc(
                         storageTy.getWidth()));
      Value qjConst = mod_arith::ConstantOp::create(b, targetLimbTy, qjAttr);
      Value reducedCj =
          mod_arith::EncapsulateOp::create(b, targetLimbTy, mrcs[j]);
      temp = mod_arith::MacOp::create(b, temp, qjConst, reducedCj);
    }
    outputLimbs.push_back(temp);
  }

  return rns::PackOp::create(b, targetBasisTy, outputLimbs).getResult();
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
