#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToSCF/CheddarGPUToSCF.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

namespace {

std::pair<int64_t, int64_t> getLimbRange(std::optional<ArrayRef<int64_t>> range,
                                         int64_t totalLimbs) {
  if (range && range->size() == 2) {
    return {(*range)[0], (*range)[1]};
  }
  return {0, totalLimbs};
}

struct ConvertBConvOp : public OpRewritePattern<BConvOp> {
  using OpRewritePattern<BConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BConvOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent module");
    auto params = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!params) return op.emitOpError("missing cheddar_gpu.params on module");

    uint32_t word = params.getWord();
    uint32_t wideWidth = 2 * word;
    int64_t degree = 1LL << params.getLogN();

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto inType = cast<MemRefType>(op.getIn().getType());
    auto tableType = cast<MemRefType>(op.getTable().getType());

    auto [outStart, outTotal] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));
    auto [inStart, inCount] =
        getLimbRange(op.getInLimbRange(), inType.getDimSize(0));

    int64_t tableDstLen = tableType.getDimSize(0);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Type intType = IntegerType::get(b.getContext(), word);
    Type wideType = IntegerType::get(b.getContext(), wideWidth);

    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value cDegree = arith::ConstantIndexOp::create(b, degree);
    Value cSrcCount = arith::ConstantIndexOp::create(b, inCount);
    Value inStartVal = arith::ConstantIndexOp::create(b, inStart);
    Value c0_2w = arith::ConstantIntOp::create(b, wideType, 0);

    std::optional<ArrayRef<int64_t>> skipRange = op.getSkipRange();
    int64_t skipStart = 0;
    int64_t skipEnd = 0;
    if (skipRange && skipRange->size() == 2) {
      skipStart = (*skipRange)[0];
      skipEnd = (*skipRange)[1];
    }

    for (int64_t m = 0; m < tableDstLen; ++m) {
      int64_t dstRelLimb = m;
      if (skipStart < skipEnd && m >= skipStart) {
        dstRelLimb += (skipEnd - skipStart);
      }
      int64_t outLimbIdx = outStart + dstRelLimb;
      uint64_t q = 0;
      if (std::optional<ArrayRef<int64_t>> dstPrimes = op.getDstPrimes();
          dstPrimes && m < static_cast<int64_t>(dstPrimes->size())) {
        q = static_cast<uint64_t>((*dstPrimes)[m]);
      } else {
        q = params.getPrimeAt(outLimbIdx);
      }

      auto modAttr = IntegerAttr::get(intType, APInt(word, q));
      auto modType = mod_arith::ModArithType::get(b.getContext(), modAttr);

      // prime_th = q * 2^word (in wideType)
      APInt qWide(wideWidth, q);
      APInt primeThVal = qWide.shl(word);
      APInt primeThHalfVal = primeThVal.lshr(1);

      Value primeTh = arith::ConstantIntOp::create(b, wideType, primeThVal);
      Value primeThHalf =
          arith::ConstantIntOp::create(b, wideType, primeThHalfVal);

      Value mVal = arith::ConstantIndexOp::create(b, m);
      Value outLimbVal = arith::ConstantIndexOp::create(b, outLimbIdx);

      // Outer loop over polynomial coefficients
      scf::ForOp::create(
          b, c0, cDegree, c1, ValueRange{},
          [&](OpBuilder& nestedBuilder, Location loc, Value xIdx,
              ValueRange /*iterArgs*/) {
            ImplicitLocOpBuilder nb(loc, nestedBuilder);

            // Inner reduction loop over source limbs s in [0, inCount)
            auto innerLoop = scf::ForOp::create(
                nb, c0, cSrcCount, c1, ValueRange{c0_2w},
                [&](OpBuilder& loopBuilder, Location loopLoc, Value sIdx,
                    ValueRange iterArgs) {
                  ImplicitLocOpBuilder lb(loopLoc, loopBuilder);
                  Value accum = iterArgs[0];

                  Value inLimbIdx = arith::AddIOp::create(lb, inStartVal, sIdx);
                  Value srcVal = memref::LoadOp::create(
                      lb, op.getIn(), ValueRange{inLimbIdx, xIdx});
                  Value tblVal = memref::LoadOp::create(lb, op.getTable(),
                                                        ValueRange{mVal, sIdx});

                  Value srcExt = arith::ExtSIOp::create(lb, wideType, srcVal);
                  Value tblExt = arith::ExtSIOp::create(lb, wideType, tblVal);
                  Value prod = arith::MulIOp::create(lb, srcExt, tblExt);
                  Value newAccum = arith::AddIOp::create(lb, accum, prod);

                  // Periodic re-centering step (value-preserving under
                  // MontReduce): if (accum < 0) accum += prime_th; if (accum >=
                  // prime_th_half) accum -= prime_th;
                  Value isNeg = arith::CmpIOp::create(
                      lb, arith::CmpIPredicate::slt, newAccum, c0_2w);
                  Value accumPlus =
                      arith::AddIOp::create(lb, newAccum, primeTh);
                  Value accumPos =
                      arith::SelectOp::create(lb, isNeg, accumPlus, newAccum);

                  Value isGeHalf = arith::CmpIOp::create(
                      lb, arith::CmpIPredicate::sge, accumPos, primeThHalf);
                  Value accumMinus =
                      arith::SubIOp::create(lb, accumPos, primeTh);
                  Value centeredAccum = arith::SelectOp::create(
                      lb, isGeHalf, accumMinus, accumPos);

                  scf::YieldOp::create(lb, ValueRange{centeredAccum});
                });

            Value finalAccum = innerLoop.getResult(0);

            // Final Montgomery reduction: (finalAccum * R^-1) mod q
            Value reducedMod =
                mod_arith::MontReduceOp::create(nb, modType, finalAccum);
            Value res = mod_arith::LiftOp::create(
                nb, intType, reducedMod, mod_arith::LiftType::STANDARD);

            memref::StoreOp::create(nb, res, op.getOut(),
                                    ValueRange{outLimbVal, xIdx});
            scf::YieldOp::create(nb);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateBConvToSCFPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertBConvOp>(patterns.getContext());
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
