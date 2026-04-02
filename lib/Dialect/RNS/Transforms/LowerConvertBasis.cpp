#include "lib/Dialect/RNS/Transforms/LowerConvertBasis.h"

#include <cstddef>
#include <utility>

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "llvm/include/llvm/ADT/APInt.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallString.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rns {

#define GEN_PASS_DEF_LOWERCONVERTBASIS
#include "lib/Dialect/RNS/Transforms/Passes.h.inc"

// Implementation based on
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
struct LowerConvertBasisOp : public OpRewritePattern<ConvertBasisOp> {
  using OpRewritePattern<ConvertBasisOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertBasisOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MLIRContext* context = op.getContext();

    Value x = op.getValue();
    auto inputBasisTy = dyn_cast<RNSType>(x.getType());
    if (!inputBasisTy) {
      return op.emitError() << "expected input basis type to be RNS";
    }
    auto targetBasisTy = dyn_cast<RNSType>(op.getTargetBasis());
    if (!targetBasisTy) {
      return op.emitError() << "expected target basis type to be RNS";
    }

    FailureOr<SmallVector<mod_arith::ModArithAttr>> qInvProdsFailable =
        buildQInvProds(context, inputBasisTy);
    if (failed(qInvProdsFailable)) {
      return op.emitError() << "failed to build mixed-radix inverse products";
    }

    // We need a map from modulus to index (on the input basis) for
    // short-circuiting
    llvm::DenseMap<APInt, size_t> inputBasisIndexByModulus;
    for (auto [i, basisTy] : llvm::enumerate(inputBasisTy.getBasisTypes())) {
      auto inputLimbTy = dyn_cast<mod_arith::ModArithType>(basisTy);
      if (!inputLimbTy) {
        return op.emitError() << "expected input basis limb to be ModArithType";
      }
      inputBasisIndexByModulus[inputLimbTy.getModulus().getValue()] = i;
    }

    ArrayRef<Type> targetBasisTypes = targetBasisTy.getBasisTypes();
    FailureOr<SmallVector<Value>> maybeMrcs =
        rns::computeMixedRadixCoeffs(b, x, *qInvProdsFailable);
    if (failed(maybeMrcs)) {
      return op.emitError()
             << "failed to compute mixed radix coefficients for input";
    }
    if (maybeMrcs->empty()) {
      return op.emitError() << "expected non-empty mixed-radix coefficients";
    }

    SmallVector<Value>& mrcs = *maybeMrcs;
    SmallVector<mod_arith::ModArithType> maInputBasisTys;
    for (auto [i, basisTy] : llvm::enumerate(inputBasisTy.getBasisTypes())) {
      auto qi = dyn_cast<mod_arith::ModArithType>(basisTy);
      if (!qi) {
        return op.emitError()
               << "expected source basis limb to be ModArithType";
      }
      APInt modulusValue = qi.getModulus().getValue();
      if (!modulusValue[0]) {
        SmallString<16> modulusStr;
        modulusValue.toStringUnsigned(modulusStr);
        return op.emitError()
               << "basis conversion requires odd moduli, "
               << "but input basis contains even modulus " << modulusStr;
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
        return op.emitError()
               << "expected target basis limb to be ModArithType";
      }
      APInt targetModulusValue = targetLimbTy.getModulus().getValue();
      if (!targetModulusValue[0]) {
        SmallString<16> modulusStr;
        targetModulusValue.toStringUnsigned(modulusStr);
        return op.emitError()
               << "basis conversion requires odd moduli, "
               << "but target basis contains even modulus " << modulusStr;
      }

      // This short-circuit optimization may not be optimal on all backends;
      // this block can be commented out while preserving correctness.
      llvm::DenseMap<APInt, size_t>::const_iterator inputIndexIt =
          inputBasisIndexByModulus.find(targetModulusValue);
      if (inputIndexIt != inputBasisIndexByModulus.end()) {
        outputLimbs.push_back(rns::ExtractResidueOp::create(
            b, x, b.getIndexAttr(inputIndexIt->second)));
        continue;
      }

      // If the output modulus isn't in the input basis, compute its
      // representative Again, this uses Horner's method with `temp` as the
      // accumulator
      Value temp =
          mod_arith::EncapsulateOp::create(b, targetLimbTy, mrcs.back());
      temp = mod_arith::ReduceOp::create(b, targetLimbTy, temp);
      for (int j = static_cast<int>(mrcs.size()) - 2; j >= 0; j--) {
        // get q_j, extend it to q_i's width, and reduce it mod q_i.
        mod_arith::ModArithType sourceLimbTy = maInputBasisTys[j];
        IntegerAttr qjAttr = IntegerAttr::get(
            storageTy, sourceLimbTy.getModulus().getValue().zextOrTrunc(
                           storageTy.getWidth()));
        Value qjConst = mod_arith::ConstantOp::create(b, targetLimbTy, qjAttr);
        Value reducedCj =
            mod_arith::EncapsulateOp::create(b, targetLimbTy, mrcs[j]);
        reducedCj = mod_arith::ReduceOp::create(b, targetLimbTy, reducedCj);
        temp = mod_arith::MacOp::create(b, temp, qjConst, reducedCj);
      }
      outputLimbs.push_back(temp);
    }

    Value result = rns::PackOp::create(b, targetBasisTy, outputLimbs);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerConvertBasis : impl::LowerConvertBasisBase<LowerConvertBasis> {
  using LowerConvertBasisBase::LowerConvertBasisBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerConvertBasisOp>(context);
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace rns
}  // namespace heir
}  // namespace mlir
