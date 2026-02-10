#ifndef LIB_DIALECT_LWE_IR_LWEOPS_H_
#define LIB_DIALECT_LWE_IR_LWEOPS_H_

#include <cstddef>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETraits.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/LWE/IR/LWEOps.h.inc"

namespace mlir {
namespace heir {
namespace lwe {

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//
inline LWECiphertextType getCtTy(Type ty) {
  return cast<LWECiphertextType>(getElementTypeOrSelf(ty));
}

inline LWECiphertextType getCtTy(Value v) { return getCtTy(v.getType()); }

inline LWEPlaintextType getPtTy(Type ty) {
  return cast<LWEPlaintextType>(getElementTypeOrSelf(ty));
}

inline LWEPlaintextType getPtTy(Value v) { return getPtTy(v.getType()); }

template <typename Op>
LogicalResult verifyMulOp(Op* op) {
  // verify dimension matches
  auto x = getCtTy(op->getLhs());
  auto y = getCtTy(op->getRhs());
  auto out = getCtTy(op->getOutput());
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
  lwe::LWECiphertextType ct;
  lwe::LWEPlaintextType pt;
  if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(op->getLhs()))) {
    ct = getCtTy(op->getLhs());
    pt = getPtTy(op->getRhs());
  } else {
    ct = getCtTy(op->getRhs());
    pt = getPtTy(op->getLhs());
  }
  auto out = getCtTy(op->getOutput());
  // verify dimension matches
  if (ct.getCiphertextSpace().getSize() != out.getCiphertextSpace().getSize()) {
    return op->emitOpError() << "output.dim == x.dim does not hold";
  }
  // verify plaintext space matches
  auto ctPlaintext = ct.getPlaintextSpace();
  auto ptPlaintext = pt.getPlaintextSpace();
  auto outPlaintext = out.getPlaintextSpace();
  if (outPlaintext != inferMulOpPlaintextSpaceAttr(op->getContext(),
                                                   ctPlaintext, ptPlaintext)) {
    return op->emitOpError() << "output plaintext space does not match";
  }
  return success();
}

template <typename Op>
LogicalResult verifyRotateOp(Op* op) {
  auto x = getCtTy(op->getInput());
  if (x.getCiphertextSpace().getSize() != 2) {
    return op->emitOpError() << "x.dim == 2 does not hold";
  }
  auto out = getCtTy(op->getOutput());
  if (out.getCiphertextSpace().getSize() != 2) {
    return op->emitOpError() << "output.dim == 2 does not hold";
  }
  return success();
}

template <typename Op>
LogicalResult verifyRelinearizeOp(Op* op) {
  auto x = getCtTy(op->getInput());
  auto out = getCtTy(op->getOutput());
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
  auto x = getCtTy(op->getInput());
  auto xRing = x.getCiphertextSpace().getRing();
  auto out = getCtTy(op->getOutput());
  auto outRing = out.getCiphertextSpace().getRing();
  if (outRing != op->getToRing()) {
    return op->emitOpError()
           << "output ciphertext_space ring should match to_ring";
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
LogicalResult verifyLevelReduceOp(Op* op) {
  auto levelToDrop = op->getLevelToDrop();
  auto x = getCtTy(op->getInput());
  auto out = getCtTy(op->getOutput());

  if (x.getModulusChain().getCurrent() < levelToDrop) {
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
  auto x = getCtTy(adaptor.getLhs());
  auto y = getCtTy(adaptor.getRhs());
  auto newDim = std::max(x.getCiphertextSpace().getSize(),
                         y.getCiphertextSpace().getSize());
  auto newCtTy = lwe::LWECiphertextType::get(
      ctx, x.getPlaintextSpace(),
      lwe::CiphertextSpaceAttr::get(ctx, x.getCiphertextSpace().getRing(),
                                    x.getCiphertextSpace().getEncryptionType(),
                                    newDim),
      x.getKey(), x.getModulusChain());

  if (auto tensorTy = dyn_cast<RankedTensorType>(adaptor.getLhs().getType())) {
    inferredReturnTypes.push_back(
        RankedTensorType::get(tensorTy.getShape(), newCtTy));
    return success();
  }

  inferredReturnTypes.push_back(newCtTy);
  return success();
}

template <typename Adaptor>
LogicalResult inferPlainOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  if (auto ct = dyn_cast<lwe::LWECiphertextType>(
          getElementTypeOrSelf(adaptor.getLhs().getType()))) {
    inferredReturnTypes.push_back(adaptor.getLhs().getType());
    return success();
  }

  if (auto ct = dyn_cast<lwe::LWECiphertextType>(
          getElementTypeOrSelf(adaptor.getRhs().getType()))) {
    inferredReturnTypes.push_back(adaptor.getRhs().getType());
    return success();
  }

  emitError(adaptor.getLhs().getLoc())
      << "expected lhs or rhs to be a ciphertext type";
  return failure();
}

template <typename Adaptor>
LogicalResult inferMulOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto x = getCtTy(adaptor.getLhs());
  auto y = getCtTy(adaptor.getRhs());
  auto newDim =
      x.getCiphertextSpace().getSize() + y.getCiphertextSpace().getSize() - 1;
  auto xPlaintextSpace = x.getPlaintextSpace();
  auto yPlaintextSpace = y.getPlaintextSpace();

  lwe::PlaintextSpaceAttr newPlaintextSpaceAttr =
      inferMulOpPlaintextSpaceAttr(ctx, xPlaintextSpace, yPlaintextSpace);

  auto newCtTy = lwe::LWECiphertextType::get(
      ctx, newPlaintextSpaceAttr,
      lwe::CiphertextSpaceAttr::get(ctx, x.getCiphertextSpace().getRing(),
                                    x.getCiphertextSpace().getEncryptionType(),
                                    newDim),
      x.getKey(), x.getModulusChain());

  if (auto tensorTy = dyn_cast<RankedTensorType>(adaptor.getLhs().getType())) {
    inferredReturnTypes.push_back(
        RankedTensorType::get(tensorTy.getShape(), newCtTy));
    return success();
  }

  inferredReturnTypes.push_back(newCtTy);
  return success();
}

template <typename Adaptor>
LogicalResult inferMulPlainOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  lwe::LWECiphertextType ct;
  lwe::LWEPlaintextType pt;
  if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(adaptor.getLhs()))) {
    ct = getCtTy(adaptor.getLhs());
    pt = getPtTy(adaptor.getRhs());
  } else {
    ct = getCtTy(adaptor.getRhs());
    pt = getPtTy(adaptor.getLhs());
  }
  auto ctPlaintextSpace = ct.getPlaintextSpace();
  auto ptPlaintextSpace = pt.getPlaintextSpace();

  lwe::PlaintextSpaceAttr newPlaintextSpaceAttr =
      inferMulOpPlaintextSpaceAttr(ctx, ctPlaintextSpace, ptPlaintextSpace);

  auto newCtTy = lwe::LWECiphertextType::get(ctx, newPlaintextSpaceAttr,
                                             ct.getCiphertextSpace(),
                                             ct.getKey(), ct.getModulusChain());

  if (auto tensorTy = dyn_cast<RankedTensorType>(adaptor.getLhs().getType())) {
    inferredReturnTypes.push_back(
        RankedTensorType::get(tensorTy.getShape(), newCtTy));
    return success();
  }

  inferredReturnTypes.push_back(newCtTy);
  return success();
}

template <typename Adaptor>
LogicalResult inferRelinearizeOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto x = getCtTy(adaptor.getInput());
  auto newCtTy = lwe::LWECiphertextType::get(
      ctx, x.getPlaintextSpace(),
      lwe::CiphertextSpaceAttr::get(ctx, x.getCiphertextSpace().getRing(),
                                    x.getCiphertextSpace().getEncryptionType(),
                                    adaptor.getToBasis().size()),
      x.getKey(), x.getModulusChain());

  if (auto tensorTy =
          dyn_cast<RankedTensorType>(adaptor.getInput().getType())) {
    inferredReturnTypes.push_back(
        RankedTensorType::get(tensorTy.getShape(), newCtTy));
    return success();
  }

  inferredReturnTypes.push_back(newCtTy);
  return success();
}

template <typename Adaptor>
LogicalResult inferLevelReduceOpReturnTypes(
    MLIRContext* ctx, Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto x = getCtTy(adaptor.getInput());
  auto levelToDrop = adaptor.getLevelToDrop();

  ModulusChainAttr newModulusChain = lwe::ModulusChainAttr::get(
      ctx, x.getModulusChain().getElements(),
      x.getModulusChain().getCurrent() - levelToDrop);
  polynomial::RingAttr newRing = getRingFromModulusChain(
      newModulusChain, x.getCiphertextSpace().getRing().getPolynomialModulus());

  auto newCtTy = lwe::LWECiphertextType::get(
      ctx, x.getPlaintextSpace(),
      lwe::CiphertextSpaceAttr::get(ctx, newRing,
                                    x.getCiphertextSpace().getEncryptionType(),
                                    x.getCiphertextSpace().getSize()),
      x.getKey(), newModulusChain);

  if (auto tensorTy =
          dyn_cast<RankedTensorType>(adaptor.getInput().getType())) {
    inferredReturnTypes.push_back(
        RankedTensorType::get(tensorTy.getShape(), newCtTy));
    return success();
  }

  inferredReturnTypes.push_back(newCtTy);
  return success();
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_IR_LWEOPS_H_
