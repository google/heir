#include "lib/Dialect/LWE/Transforms/DecomposeLWEOps.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/APIntUtils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

namespace {

FailureOr<Value> createPInvModQConstant(
    ModDownOp op, ImplicitLocOpBuilder& b, rns::RNSType inputRNSType,
    ArrayRef<mod_arith::ModArithType> keySwitchPrimeTypes) {
  SmallVector<Attribute> pInvModQResidues;
  pInvModQResidues.reserve(inputRNSType.getBasisTypes().size());

  for (Type ty : inputRNSType.getBasisTypes()) {
    auto qiTy = dyn_cast<mod_arith::ModArithType>(ty);
    if (!qiTy) {
      op.emitOpError()
          << "input basis element must be a mod_arith type to build P^-1 mod Q";
      return failure();
    }

    APInt qi = qiTy.getModulus().getValue();
    APInt pModQi(qi.getBitWidth(), 1);
    for (mod_arith::ModArithType pjTy : keySwitchPrimeTypes) {
      APInt pj = pjTy.getModulus().getValue();
      unsigned reductionWidth = std::max(qi.getBitWidth(), pj.getBitWidth());
      APInt pjModQi = pj.zextOrTrunc(reductionWidth)
                          .urem(qi.zextOrTrunc(reductionWidth))
                          .zextOrTrunc(qi.getBitWidth());
      pModQi = modularMultiplication(pModQi, pjModQi, qi);
    }

    APInt pInvModQi = multiplicativeInverse(pModQi, qi);
    if (pInvModQi.isZero()) {
      op.emitOpError() << "failed to compute P^-1 modulo input limb " << qiTy;
      return failure();
    }

    IntegerAttr pInvAttr =
        IntegerAttr::get(qiTy.getModulus().getType(), pInvModQi);
    pInvModQResidues.push_back(
        mod_arith::ModArithAttr::get(op.getContext(), qiTy, pInvAttr));
  }

  auto pInvModQAttr = rns::RNSAttr::get(inputRNSType, pInvModQResidues);
  return rns::ConstantOp::create(b, inputRNSType, pInvModQAttr).getResult();
}

}  // namespace

#define GEN_PASS_DEF_DECOMPOSELWEOPS
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

LogicalResult DecomposeModDownPattern::matchAndRewrite(
    ModDownOp op, PatternRewriter& rewriter) const {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  lwe::LWECiphertextType ctType = op.getInput().getType();
  auto eltTy = dyn_cast<rns::RNSType>(
      ctType.getCiphertextSpace().getRing().getCoefficientType());
  if (!eltTy) return op->emitOpError("Expected element type to be RNSType");
  ArrayRef<Type> inputBasisTypes = eltTy.getBasisTypes();
  int64_t numInputLimbs = inputBasisTypes.size();

  auto targetBasisTy = dyn_cast<rns::RNSType>(op.getTargetBasis());
  if (!targetBasisTy) return failure();
  ArrayRef<Type> targetBasisTypes = targetBasisTy.getBasisTypes();
  int64_t numTargetLimbs = targetBasisTypes.size();

  SmallVector<mod_arith::ModArithType> keySwitchPrimeTypes;
  for (Type ty : inputBasisTypes.drop_front(targetBasisTypes.size())) {
    auto modArithType = dyn_cast<mod_arith::ModArithType>(ty);
    if (!modArithType) {
      return op->emitOpError()
             << "key-switching prime basis element must be a mod_arith type";
    }
    keySwitchPrimeTypes.push_back(modArithType);
  }

  SmallVector<Value> basePrimeCoeffs;
  SmallVector<Value> convertedKeySwitchPrimeCoeffs;
  int64_t numCiphertextComponents = ctType.getCiphertextSpace().getSize();
  for (int64_t i = 0; i < numCiphertextComponents; ++i) {
    Value coeff =
        lwe::ExtractCoeffOp::create(b, op.getInput(), b.getIndexAttr(i));

    // Extract the mod-q components.
    Value basePrimeSlice = lwe::ExtractSliceOp::create(
        b, coeff, b.getIndexAttr(0), b.getIndexAttr(numTargetLimbs));
    basePrimeCoeffs.push_back(basePrimeSlice);

    // Extract the key-switch-prime components and basis-convert them to mod-q.
    Value kskPrimeSlice = lwe::ExtractSliceOp::create(
        b, coeff, b.getIndexAttr(numTargetLimbs),
        b.getIndexAttr(numInputLimbs - numTargetLimbs));
    Value kskSliceToBasePrimes = lwe::ConvertBasisOp::create(
        b, kskPrimeSlice, TypeAttr::get(targetBasisTy));
    convertedKeySwitchPrimeCoeffs.push_back(kskSliceToBasePrimes);
  }

  auto outputCtType = cast<lwe::LWECiphertextType>(op.getOutput().getType());
  Value basePrimeCt =
      lwe::FromCoeffsOp::create(b, outputCtType, basePrimeCoeffs);
  Value kskPrimeCt =
      lwe::FromCoeffsOp::create(b, outputCtType, convertedKeySwitchPrimeCoeffs);

  // subtract
  Value diffCt = lwe::RSubOp::create(b, basePrimeCt, kskPrimeCt);

  // scale by a constant, which is P^{-1} mod Q,
  // where P is the product of the special/key-switch primes and Q is the
  // product of the input basis
  FailureOr<Value> pInvModQ =
      createPInvModQConstant(op, b, targetBasisTy, keySwitchPrimeTypes);
  if (failed(pInvModQ)) return failure();
  Value scaledCt = lwe::MulScalarOp::create(b, diffCt, pInvModQ.value());

  rewriter.replaceOp(op, scaledCt);
  return success();
}

LogicalResult DecomposeKeySwitchPattern::matchAndRewrite(
    KeySwitchInnerOp op, PatternRewriter& rewriter) const {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  auto ringEltType = cast<lwe::LWERingEltType>(op.getValue().getType());
  auto inputRNSType =
      dyn_cast<rns::RNSType>(ringEltType.getRing().getCoefficientType());
  if (!inputRNSType) {
    return op->emitOpError() << "input type must be an RNS ring element";
  }

  RankedTensorType keyTensorType = op.getKeySwitchingKey().getType();
  auto keyCtType =
      dyn_cast<lwe::LWECiphertextType>(keyTensorType.getElementType());
  if (!keyCtType) {
    return op->emitOpError()
           << "key-switching key must be a tensor of ciphertexts";
  }
  auto keyRNSType = dyn_cast<rns::RNSType>(
      keyCtType.getCiphertextSpace().getRing().getCoefficientType());
  if (!keyRNSType) {
    return op->emitOpError()
           << "key-switching key must have an RNS coefficient type";
  }

  ArrayRef<Type> inputBasis = inputRNSType.getBasisTypes();
  ArrayRef<Type> keyBasis = keyRNSType.getBasisTypes();
  if (keyBasis.size() <= inputBasis.size()) {
    return op->emitOpError()
           << "key-switching key basis must contain the input basis plus "
              "key-switch primes";
  }
  if (!std::equal(inputBasis.begin(), inputBasis.end(), keyBasis.begin())) {
    return op->emitOpError()
           << "key-switching key basis must start with the input basis";
  }

  SmallVector<mod_arith::ModArithType> keySwitchPrimeTypes;
  for (Type ty : keyBasis.drop_front(inputBasis.size())) {
    auto modArithType = dyn_cast<mod_arith::ModArithType>(ty);
    if (!modArithType) {
      return op->emitOpError()
             << "key-switching prime basis element must be a mod_arith type";
    }
    keySwitchPrimeTypes.push_back(modArithType);
  }

  int64_t partSize = keySwitchPrimeTypes.size();
  if (!partSize) {
    return rewriter.notifyMatchFailure(
        op, "Cannot lower key_switch_inner with empty modulus chain");
  }

  // #############################
  // ## Step 1: Decompose Input ##
  // #############################

  int rnsLength = inputBasis.size();
  int64_t numFullPartitions = rnsLength / partSize;
  int64_t extraPartStart = partSize * numFullPartitions;
  int64_t extraPartSize = rnsLength - extraPartStart;
  SmallVector<Value> partitions;
  for (int i = 0; i < numFullPartitions; ++i) {
    partitions.push_back(lwe::ExtractSliceOp::create(
        b, op.getValue(), b.getIndexAttr(i * partSize),
        b.getIndexAttr(partSize)));
  }

  // Partition the RNS limbs of the input ring element into parts
  // according to the number of special (key switch) primes in the parameters.
  //
  // Since the number of special primes may not evenly divide the number of
  // RNS limbs, there may be an extra part containing the trailing size.
  //
  // Since each part also has a different RNS type, we can't group them
  // together into a tensor and have to deal with the individual SSA values in
  // an unrolled manner.
  //
  // The input will be a ringelt<RNS>, where RNS is the RNS type of the
  // ring element. The result after this part is a list of Value of types
  // ringelt<R1>, ringelt<R2>, ...
  if (extraPartSize > 0) {
    partitions.push_back(lwe::ExtractSliceOp::create(
        b, op.getValue(), b.getIndexAttr(extraPartStart),
        b.getIndexAttr(extraPartSize)));
  }
  int64_t kskLen = op.getKeySwitchingKey().getType().getShape()[0];
  if (kskLen != static_cast<int64_t>(partitions.size())) {
    return op->emitOpError()
           << "KeySwitchingKey must have shape " << partitions.size()
           << "xRNS, but it has shape " << kskLen << "xRNS";
  }

  // ###########################################
  // ## Step 2: Extend the basis of each part ##
  // ###########################################
  // The RNS type of each part corresponds to a different subset of the
  // input's RNS type. Note that the input's RNS type may be a subset of
  // scheme's full modulus chain, e.g., if there was a rescale in the IR
  // before this op.
  //
  // We want to extend these RNS types to the RNS type which is the input's
  // RNS type, plus the key-switching primes. We assume that the key-switching
  // key has this RNS basis, which we verify below.
  SmallVector<Type> extModuli;
  for (auto ty : inputRNSType.getBasisTypes()) {
    extModuli.push_back(ty);
  }
  for (auto primeType : keySwitchPrimeTypes) {
    extModuli.push_back(primeType);
  }
  rns::RNSType newBasisType = rns::RNSType::get(op.getContext(), extModuli);

  // Now apply basis extension
  SmallVector<Value> extendedPartitions;
  for (auto value : partitions) {
    extendedPartitions.push_back(
        lwe::ConvertBasisOp::create(b, value, TypeAttr::get(newBasisType)));
  }

  // ###########################################
  // ## Step 3: Compute dot product with KSKs ##
  // ###########################################
  // KSK is a K x CT
  // extendedPartitions is a K-sized list of RingElts
  int k = extendedPartitions.size();
  Value sum;
  for (int i = 0; i < k; i++) {
    Value idx = arith::ConstantIndexOp::create(b, i).getResult();
    Value ksk = tensor::ExtractOp::create(b, op.getKeySwitchingKey(), idx);
    Value prod = lwe::RMulRingEltOp::create(b, extendedPartitions[i], ksk);
    if (i == 0) {
      sum = prod;
    } else {
      sum = lwe::RAddOp::create(b, sum, prod);
    }
  }

  // ##########################################
  // ## Step 4: Remove the key-switch primes ##
  // ##########################################
  Value scaledCt = lwe::ModDownOp::create(b, sum, TypeAttr::get(inputRNSType));
  Value newConstTerm =
      lwe::ExtractCoeffOp::create(b, scaledCt, b.getIndexAttr(0));
  Value newLinearTerm =
      lwe::ExtractCoeffOp::create(b, scaledCt, b.getIndexAttr(1));

  rewriter.replaceOp(op, {newConstTerm, newLinearTerm});
  return success();
}

struct DecomposeLWEOps : impl::DecomposeLWEOpsBase<DecomposeLWEOps> {
  using DecomposeLWEOpsBase::DecomposeLWEOpsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<DecomposeKeySwitchPattern, DecomposeModDownPattern>(context);
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
