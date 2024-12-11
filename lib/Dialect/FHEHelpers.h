#ifndef LIB_DIALECT_FHEHELPERS_H_
#define LIB_DIALECT_FHEHELPERS_H_

#include <cstddef>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Refactored out verifiers and inferReturnTypes for BGV and CKKS operations.
template <typename Op>
LogicalResult verifyMulOp(Op* op) {
  auto x = op->getLhs().getType();
  auto y = op->getRhs().getType();
  if (x.getRlweParams().getDimension() != y.getRlweParams().getDimension()) {
    return op->emitOpError() << "input dimensions do not match";
  }
  auto out = op->getOutput().getType();
  if (out.getRlweParams().getDimension() !=
      y.getRlweParams().getDimension() + x.getRlweParams().getDimension() - 1) {
    return op->emitOpError() << "output.dim == x.dim + y.dim - 1 does not hold";
  }
  return success();
}

template <typename Op>
LogicalResult verifyRotateOp(Op* op) {
  auto x = op->getInput().getType();
  if (x.getRlweParams().getDimension() != 2) {
    return op->emitOpError() << "x.dim == 2 does not hold";
  }
  auto out = op->getOutput().getType();
  if (out.getRlweParams().getDimension() != 2) {
    return op->emitOpError() << "output.dim == 2 does not hold";
  }
  return success();
}

template <typename Op>
LogicalResult verifyRelinearizeOp(Op* op) {
  auto x = op->getInput().getType();
  auto out = op->getOutput().getType();
  if (x.getRlweParams().getDimension() != op->getFromBasis().size()) {
    return op->emitOpError() << "input dimension does not match from_basis";
  }
  if (out.getRlweParams().getDimension() != op->getToBasis().size()) {
    return op->emitOpError() << "output dimension does not match to_basis";
  }
  return success();
}

template <typename Op>
LogicalResult verifyModulusSwitchOrRescaleOp(Op* op) {
  auto x = op->getInput().getType();
  auto xRing = x.getRlweParams().getRing();

  auto out = op->getOutput().getType();
  auto outRing = out.getRlweParams().getRing();
  if (outRing != op->getToRing()) {
    return op->emitOpError() << "output ring should match to_ring";
  }

  bool isModArith = false;
  bool isRNS = false;

  auto xRingCoeffType =
      dyn_cast<mod_arith::ModArithType>(xRing.getCoefficientType());
  auto outRingCoeffType =
      dyn_cast<mod_arith::ModArithType>(outRing.getCoefficientType());

  if (xRingCoeffType && outRingCoeffType) {
    isModArith = true;
    if (xRingCoeffType.getModulus().getValue().ule(
            outRingCoeffType.getModulus().getValue())) {
      return op->emitOpError() << "output ring modulus should be less than the "
                                  "input ring modulus";
    }
    if (!xRingCoeffType.getModulus()
             .getValue()
             .urem(outRingCoeffType.getModulus().getValue())
             .isZero()) {
      return op->emitOpError()
             << "output ring modulus should divide the input ring modulus";
    }
  }

  auto xRNSRingCoeffType = dyn_cast<rns::RNSType>(xRing.getCoefficientType());
  auto outRNSRingCoeffType =
      dyn_cast<rns::RNSType>(outRing.getCoefficientType());

  if (xRNSRingCoeffType && outRNSRingCoeffType) {
    isRNS = true;

    auto xBasis = xRNSRingCoeffType.getBasisTypes();
    auto outBasis = outRNSRingCoeffType.getBasisTypes();

    if (xBasis.size() <= outBasis.size()) {
      return op->emitOpError()
             << "output ring basis size should be less than the "
                "input ring basis size";
    }

    for (size_t i = 0; i < outBasis.size(); ++i) {
      if (xBasis[i] != outBasis[i]) {
        return op->emitOpError() << "output ring basis should be a prefix of "
                                    "the input ring basis";
      }
    }
  }

  if (!isModArith && !isRNS) {
    return op->emitOpError() << "input and output rings should have "
                                "either mod_arith or rns coefficient types";
  }

  return success();
}

template <typename Adaptor>
LogicalResult inferMulOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto x = cast<lwe::RLWECiphertextType>(adaptor.getLhs().getType());
  auto y = cast<lwe::RLWECiphertextType>(adaptor.getRhs().getType());
  auto newDim =
      x.getRlweParams().getDimension() + y.getRlweParams().getDimension() - 1;
  inferredReturnTypes.push_back(lwe::RLWECiphertextType::get(
      ctx, x.getEncoding(),
      lwe::RLWEParamsAttr::get(ctx, newDim, x.getRlweParams().getRing()),
      x.getUnderlyingType()));
  return success();
}

template <typename Adaptor>
LogicalResult inferRelinearizeOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto x = cast<lwe::RLWECiphertextType>(adaptor.getInput().getType());
  inferredReturnTypes.push_back(lwe::RLWECiphertextType::get(
      ctx, x.getEncoding(),
      lwe::RLWEParamsAttr::get(ctx, 2, x.getRlweParams().getRing()),
      x.getUnderlyingType()));
  return success();
}

template <typename Op>
LogicalResult verifyExtractOp(Op* op) {
  auto inputTy = op->getInput().getType();
  auto tensorTy = dyn_cast<RankedTensorType>(inputTy.getUnderlyingType());
  if (!tensorTy) {
    return op->emitOpError() << "input RLWE ciphertext type must have a ranked "
                                "tensor as its underlying_type, but found "
                             << inputTy.getUnderlyingType();
  }

  auto outputScalarType = op->getOutput().getType().getUnderlyingType();
  if (tensorTy.getElementType() != outputScalarType) {
    return op->emitOpError()
           << "output RLWE ciphertext's underlying_type must be "
              "the element type of the input ciphertext's "
              "underlying tensor type, but found tensor type "
           << tensorTy << " and output type " << outputScalarType;
  }
  return success();
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_FHEHELPERS_H_
