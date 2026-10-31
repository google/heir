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

struct ConvertModDownEpilogueOp : public OpRewritePattern<ModDownEpilogueOp> {
  using OpRewritePattern<ModDownEpilogueOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModDownEpilogueOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent module");
    auto params = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!params) return op.emitOpError("missing cheddar_gpu.params on module");

    uint32_t word = params.getWord();
    int64_t degree = 1LL << params.getLogN();

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t numLimbs = outType.getDimSize(0);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Type intType = IntegerType::get(b.getContext(), word);

    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value cDegree = arith::ConstantIndexOp::create(b, degree);

    int64_t padStart = op.getPadStart();
    int64_t padEnd = op.getPadEnd();
    Value padStartVal = arith::ConstantIndexOp::create(b, padStart);

    bool hasSrc2 = (op.getSrc2() != nullptr);
    bool hasPadding = (op.getSrc2Padding() != nullptr);

    for (int64_t l = 0; l < numLimbs; ++l) {
      uint64_t q = params.getPrimeAt(l);
      auto modAttr = IntegerAttr::get(intType, APInt(word, q));
      auto modType = mod_arith::ModArithType::get(b.getContext(), modAttr);

      Value lVal = arith::ConstantIndexOp::create(b, l);
      Value invPProdVal =
          memref::LoadOp::create(b, op.getInvPrimeProd(), ValueRange{lVal});
      Value invPProdMod =
          mod_arith::EncapsulateOp::create(b, modType, invPProdVal);

      bool limbHasSrc2 = hasSrc2 && (l >= padStart && l < padEnd);

      // Loop over coefficients
      scf::ForOp::create(
          b, c0, cDegree, c1, ValueRange{},
          [&](OpBuilder& nestedBuilder, Location loc, Value xIdx,
              ValueRange /*iterArgs*/) {
            ImplicitLocOpBuilder nb(loc, nestedBuilder);

            Value term1Mod;
            if (limbHasSrc2) {
              Value src2LimbIdx = arith::SubIOp::create(nb, lVal, padStartVal);
              Value s2Val = memref::LoadOp::create(
                  nb, op.getSrc2(), ValueRange{src2LimbIdx, xIdx});
              Value s2Mod =
                  mod_arith::EncapsulateOp::create(nb, modType, s2Val);

              if (hasPadding) {
                Value padVal = memref::LoadOp::create(nb, op.getSrc2Padding(),
                                                      ValueRange{src2LimbIdx});
                Value padMod =
                    mod_arith::EncapsulateOp::create(nb, modType, padVal);
                term1Mod =
                    mod_arith::MontMulOp::create(nb, modType, s2Mod, padMod);
              } else {
                term1Mod = s2Mod;
              }
            } else {
              Value zeroVal = arith::ConstantIntOp::create(nb, intType, 0);
              term1Mod = mod_arith::EncapsulateOp::create(nb, modType, zeroVal);
            }

            Value nttVal = memref::LoadOp::create(nb, op.getNttBconv(),
                                                  ValueRange{lVal, xIdx});
            Value nttMod =
                mod_arith::EncapsulateOp::create(nb, modType, nttVal);

            Value diffMod =
                mod_arith::SubOp::create(nb, modType, term1Mod, nttMod);
            Value resMod =
                mod_arith::MontMulOp::create(nb, modType, diffMod, invPProdMod);

            Value res = mod_arith::LiftOp::create(
                nb, intType, resMod, mod_arith::LiftType::STANDARD);
            memref::StoreOp::create(nb, res, op.getOut(),
                                    ValueRange{lVal, xIdx});
            scf::YieldOp::create(nb);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertNTTModDownOp : public OpRewritePattern<NTTModDownOp> {
  using OpRewritePattern<NTTModDownOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NTTModDownOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto tempType = cast<MemRefType>(op.getNttIn().getType());
    Value nttTemp = memref::AllocOp::create(b, tempType);

    DenseI64ArrayAttr primeRangeAttr;
    if (!op.getPrimeIndices() && op.getPrimeOffset() > 0) {
      primeRangeAttr = b.getDenseI64ArrayAttr(
          {static_cast<int64_t>(op.getPrimeOffset()), tempType.getDimSize(0)});
    }

    NTTOp::create(b, nttTemp, op.getNttIn(),
                  /*skip_copy_src=*/Value(),
                  FormAttr::get(b.getContext(), Form::Coeff),
                  FormAttr::get(b.getContext(), Form::Eval),
                  /*montgomery_conversion=*/b.getBoolAttr(false),
                  /*skip_range=*/DenseI64ArrayAttr(),
                  /*in_limb_range=*/DenseI64ArrayAttr(),
                  /*out_limb_range=*/DenseI64ArrayAttr(), primeRangeAttr,
                  op.getPrimeIndicesAttr(),
                  /*skip_copy_in_offset=*/0);

    Value src2 = op.getSrc2();
    if (src2 && op.getSrc2Offset() > 0) {
      int64_t padLen = op.getPadEnd() - op.getPadStart();
      int64_t degree = tempType.getDimSize(1);
      Type intType = tempType.getElementType();
      auto layout = StridedLayoutAttr::get(
          b.getContext(), /*offset=*/op.getSrc2Offset() * degree,
          /*strides=*/{degree, 1});
      auto subviewType = MemRefType::get({padLen, degree}, intType, layout);
      SmallVector<OpFoldResult> offsets = {b.getIndexAttr(op.getSrc2Offset()),
                                           b.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes = {b.getIndexAttr(padLen),
                                         b.getIndexAttr(degree)};
      SmallVector<OpFoldResult> strides = {b.getIndexAttr(1),
                                           b.getIndexAttr(1)};
      src2 = memref::SubViewOp::create(b, subviewType, op.getSrc2(), offsets,
                                       sizes, strides);
    }

    ModDownEpilogueOp::create(b, op.getOut(), nttTemp, op.getInvPrimeProd(),
                              src2, op.getSrc2Padding(), op.getPadStart(),
                              op.getPadEnd(), Form::Eval, Form::Eval);

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateModDownEpilogueToSCFPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertModDownEpilogueOp, ConvertNTTModDownOp>(
      patterns.getContext());
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
