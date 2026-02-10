#include "lib/Dialect/CKKS/Transforms/DecomposeKeySwitch.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Parameters/CKKS/Params.h"
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
namespace ckks {

#define GEN_PASS_DEF_DECOMPOSEKEYSWITCH
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

LogicalResult DecomposeKeySwitchPattern::matchAndRewrite(
    KeySwitchInnerOp op, PatternRewriter& rewriter) const {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  SchemeParamAttr schemeParamAttr =
      op->getParentOfType<ModuleOp>()->getAttrOfType<SchemeParamAttr>(
          CKKSDialect::kSchemeParamAttrName);
  if (!schemeParamAttr) {
    return op->emitOpError()
           << "Cannot find scheme param attribute on parent module";
  }
  auto schemeParam = getSchemeParamFromAttr(schemeParamAttr);

  int64_t partSize = schemeParam.getPi().size();
  if (!partSize) {
    return rewriter.notifyMatchFailure(
        op, "Cannot lower key_switch_inner with empty modulus chain");
  }

  // #############################
  // ## Step 1: Decompose Input ##
  // #############################

  auto ringEltType = cast<lwe::LWERingEltType>(op.getValue().getType());
  auto inputRNSType =
      cast<rns::RNSType>(ringEltType.getRing().getCoefficientType());
  if (!inputRNSType) {
    return rewriter.notifyMatchFailure(
        op, "Input type must be an RNS ring element");
  }

  int rnsLength = inputRNSType.getBasisTypes().size();
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
  for (auto prime : schemeParam.getPi()) {
    extModuli.push_back(mod_arith::ModArithType::get(
        op.getContext(), b.getI64IntegerAttr(prime)));
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
  Value constTerm = lwe::ExtractCoeffOp::create(b, sum, b.getIndexAttr(0));
  Value linearTerm = lwe::ExtractCoeffOp::create(b, sum, b.getIndexAttr(1));
  Value modDownConstTerm =
      lwe::ConvertBasisOp::create(b, constTerm, TypeAttr::get(inputRNSType));
  Value modDownLinearTerm =
      lwe::ConvertBasisOp::create(b, linearTerm, TypeAttr::get(inputRNSType));

  rewriter.replaceOp(op, {modDownConstTerm, modDownLinearTerm});
  return success();
}

struct DecomposeKeySwitch : impl::DecomposeKeySwitchBase<DecomposeKeySwitch> {
  using DecomposeKeySwitchBase::DecomposeKeySwitchBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<DecomposeKeySwitchPattern>(context);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
