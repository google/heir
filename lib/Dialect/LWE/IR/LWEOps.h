#ifndef LIB_DIALECT_LWE_IR_LWEOPS_H_
#define LIB_DIALECT_LWE_IR_LWEOPS_H_

#include <cstddef>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETraits.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/LWE/IR/LWEOps.h.inc"

namespace mlir {
namespace heir {
namespace lwe {

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

template <typename Op>
LogicalResult verifyMulOp(Op* op) {
  // verify dimension matches
  auto x = op->getLhs().getType();
  auto y = op->getRhs().getType();
  if (x.getCiphertextSpace().getSize() != y.getCiphertextSpace().getSize()) {
    return op->emitOpError() << "input dimensions do not match";
  }
  auto out = op->getOutput().getType();
  if (out.getCiphertextSpace().getSize() !=
      y.getCiphertextSpace().getSize() + x.getCiphertextSpace().getSize() - 1) {
    return op->emitOpError() << "output.dim == x.dim + y.dim - 1 does not hold";
  }
  // verify plaintext space matches
  auto xPlaintext = x.getPlaintextSpace();
  auto yPlaintext = y.getPlaintextSpace();
  auto outPlaintext = out.getPlaintextSpace();
  if (outPlaintext !=
      inferMulOpPlaintextSpaceAttr(op->getContext(), xPlaintext, yPlaintext)) {
    return op->emitOpError() << "output plaintext space does not match";
  }
  return success();
}

template <typename Op>
LogicalResult verifyMulPlainOp(Op* op) {
  auto x = op->getCiphertextInput().getType();
  auto y = op->getPlaintextInput().getType();
  auto out = op->getOutput().getType();
  // verify dimension matches
  if (x.getCiphertextSpace().getSize() != out.getCiphertextSpace().getSize()) {
    return op->emitOpError() << "output.dim == x.dim does not hold";
  }
  // verify plaintext space matches
  auto xPlaintext = x.getPlaintextSpace();
  auto yPlaintext = y.getPlaintextSpace();
  auto outPlaintext = out.getPlaintextSpace();
  if (outPlaintext !=
      inferMulOpPlaintextSpaceAttr(op->getContext(), xPlaintext, yPlaintext)) {
    return op->emitOpError() << "output plaintext space does not match";
  }
  return success();
}

template <typename Op>
LogicalResult verifyRotateOp(Op* op) {
  auto x = op->getInput().getType();
  if (x.getCiphertextSpace().getSize() != 2) {
    return op->emitOpError() << "x.dim == 2 does not hold";
  }
  auto out = op->getOutput().getType();
  if (out.getCiphertextSpace().getSize() != 2) {
    return op->emitOpError() << "output.dim == 2 does not hold";
  }
  return success();
}

template <typename Op>
LogicalResult verifyRelinearizeOp(Op* op) {
  auto x = op->getInput().getType();
  auto out = op->getOutput().getType();
  if (x.getCiphertextSpace().getSize() != op->getFromBasis().size()) {
    return op->emitOpError() << "input dimension does not match from_basis";
  }
  if (out.getCiphertextSpace().getSize() != op->getToBasis().size()) {
    return op->emitOpError() << "output dimension does not match to_basis";
  }
  return success();
}

template <typename Op>
LogicalResult verifyModulusSwitchOrRescaleOp(Op* op) {
  auto x = op->getInput().getType();
  auto xRing = x.getCiphertextSpace().getRing();

  auto out = op->getOutput().getType();
  auto outRing = out.getCiphertextSpace().getRing();
  if (outRing != op->getToRing()) {
    return op->emitOpError() << "output ring should match to_ring";
  }

  auto outPlaintextSpace = out.getPlaintextSpace();
  auto xPlaintextSpace = x.getPlaintextSpace();

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

    auto dividedModulus = xRingCoeffType.getModulus().getValue().sdiv(
        outRingCoeffType.getModulus().getValue());
    auto newPlaintextSpace = inferModulusSwitchOrRescaleOpPlaintextSpaceAttr(
        op->getContext(), xPlaintextSpace, dividedModulus);
    if (outPlaintextSpace != newPlaintextSpace) {
      return op->emitOpError() << "output plaintext space does not match";
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

    APInt dividedModulus = APInt(64, 1);
    for (size_t i = outBasis.size(); i < xBasis.size(); ++i) {
      auto currentModulus =
          cast<mod_arith::ModArithType>(xBasis[i]).getModulus().getValue();
      auto newWidth =
          dividedModulus.getBitWidth() + currentModulus.getBitWidth();
      dividedModulus = dividedModulus.zext(newWidth);
      currentModulus = currentModulus.zext(newWidth);
      dividedModulus *= currentModulus;
    }
    auto newPlaintextSpace = inferModulusSwitchOrRescaleOpPlaintextSpaceAttr(
        op->getContext(), xPlaintextSpace, dividedModulus);
    if (outPlaintextSpace != newPlaintextSpace) {
      return op->emitOpError() << "output plaintext space does not match";
    }
  }

  if (!isModArith && !isRNS) {
    return op->emitOpError() << "input and output rings should have "
                                "either mod_arith or rns coefficient types";
  }

  return success();
}

template <typename Op>
LogicalResult verifyExtractOp(Op* op) {
  auto inputTy = op->getInput().getType();
  auto tensorTy =
      dyn_cast<RankedTensorType>(inputTy.getApplicationData().getMessageType());
  if (!tensorTy) {
    return op->emitOpError() << "input RLWE ciphertext type must have a ranked "
                                "tensor as its underlying_type, but found "
                             << inputTy.getApplicationData().getMessageType();
  }

  auto outputScalarType =
      op->getOutput().getType().getApplicationData().getMessageType();
  if (tensorTy.getElementType() != outputScalarType) {
    return op->emitOpError()
           << "output RLWE ciphertext's underlying_type must be "
              "the element type of the input ciphertext's "
              "underlying tensor type, but found tensor type "
           << tensorTy << " and output type " << outputScalarType;
  }
  return success();
}

template <typename Op>
LogicalResult verifyLevelReduceOp(Op* op) {
  auto levelToDrop = op->getLevelToDrop();
  auto x = op->getInput().getType();
  auto out = op->getOutput().getType();

  if (x.getModulusChain().getCurrent() <= levelToDrop) {
    return op->emitOpError() << "level_to_drop is out of bounds";
  }

  if (out.getModulusChain().getCurrent() !=
      x.getModulusChain().getCurrent() - levelToDrop) {
    return op->emitOpError() << "output modulus chain size does not match";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Op type inference.
//===----------------------------------------------------------------------===//

template <typename Adaptor>
LogicalResult inferAddOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto x = cast<lwe::NewLWECiphertextType>(adaptor.getLhs().getType());
  auto y = cast<lwe::NewLWECiphertextType>(adaptor.getRhs().getType());
  auto newDim = std::max(x.getCiphertextSpace().getSize(),
                         y.getCiphertextSpace().getSize());
  inferredReturnTypes.push_back(lwe::NewLWECiphertextType::get(
      ctx, x.getApplicationData(), x.getPlaintextSpace(),
      lwe::CiphertextSpaceAttr::get(ctx, x.getCiphertextSpace().getRing(),
                                    x.getCiphertextSpace().getEncryptionType(),
                                    newDim),
      x.getKey(), x.getModulusChain()));
  return success();
}

template <typename Adaptor>
LogicalResult inferMulOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto x = cast<lwe::NewLWECiphertextType>(adaptor.getLhs().getType());
  auto y = cast<lwe::NewLWECiphertextType>(adaptor.getRhs().getType());
  auto newDim =
      x.getCiphertextSpace().getSize() + y.getCiphertextSpace().getSize() - 1;
  auto xPlaintextSpace = x.getPlaintextSpace();
  auto yPlaintextSpace = y.getPlaintextSpace();

  lwe::PlaintextSpaceAttr newPlaintextSpaceAttr =
      inferMulOpPlaintextSpaceAttr(ctx, xPlaintextSpace, yPlaintextSpace);

  inferredReturnTypes.push_back(lwe::NewLWECiphertextType::get(
      ctx, x.getApplicationData(), newPlaintextSpaceAttr,
      lwe::CiphertextSpaceAttr::get(ctx, x.getCiphertextSpace().getRing(),
                                    x.getCiphertextSpace().getEncryptionType(),
                                    newDim),
      x.getKey(), x.getModulusChain()));
  return success();
}

template <typename Adaptor>
LogicalResult inferMulPlainOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto x =
      cast<lwe::NewLWECiphertextType>(adaptor.getCiphertextInput().getType());
  auto y =
      cast<lwe::NewLWEPlaintextType>(adaptor.getPlaintextInput().getType());
  auto xPlaintextSpace = x.getPlaintextSpace();
  auto yPlaintextSpace = y.getPlaintextSpace();

  lwe::PlaintextSpaceAttr newPlaintextSpaceAttr =
      inferMulOpPlaintextSpaceAttr(ctx, xPlaintextSpace, yPlaintextSpace);

  inferredReturnTypes.push_back(lwe::NewLWECiphertextType::get(
      ctx, x.getApplicationData(), newPlaintextSpaceAttr,
      x.getCiphertextSpace(), x.getKey(), x.getModulusChain()));
  return success();
}

template <typename Adaptor>
LogicalResult inferRelinearizeOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto x = cast<lwe::NewLWECiphertextType>(adaptor.getInput().getType());
  inferredReturnTypes.push_back(lwe::NewLWECiphertextType::get(
      ctx, x.getApplicationData(), x.getPlaintextSpace(),
      lwe::CiphertextSpaceAttr::get(ctx, x.getCiphertextSpace().getRing(),
                                    x.getCiphertextSpace().getEncryptionType(),
                                    2),
      x.getKey(), x.getModulusChain()));
  return success();
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_IR_LWEOPS_H_
