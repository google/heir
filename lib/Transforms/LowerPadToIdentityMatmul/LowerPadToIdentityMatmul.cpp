#include "lib/Transforms/LowerPadToIdentityMatmul/LowerPadToIdentityMatmul.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LOWERPADTOIDENTITYMATMUL
#include "lib/Transforms/LowerPadToIdentityMatmul/LowerPadToIdentityMatmul.h.inc"

namespace {

Value createIdentityConstant(PatternRewriter& rewriter, Location loc,
                             Type elemType, ArrayRef<int64_t> shape,
                             int64_t rowDim, int64_t colDim) {
  auto shapedType = RankedTensorType::get(shape, elemType);
  int64_t totalElements = shapedType.getNumElements();

  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    SmallVector<APFloat> values;
    values.reserve(totalElements);
    APFloat zeroVal(floatType.getFloatSemantics(), 0);
    APFloat oneVal(floatType.getFloatSemantics(), 1);

    if (shape.size() == 2) {
      int64_t rows = shape[0];
      int64_t cols = shape[1];
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          values.push_back((r == c) ? oneVal : zeroVal);
        }
      }
    } else if (shape.size() == 3) {
      int64_t batch = shape[0];
      int64_t rows = shape[1];
      int64_t cols = shape[2];
      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < rows; ++r) {
          for (int64_t c = 0; c < cols; ++c) {
            values.push_back((r == c) ? oneVal : zeroVal);
          }
        }
      }
    }
    auto attr = DenseElementsAttr::get(shapedType, values);
    return arith::ConstantOp::create(rewriter, loc, shapedType, attr);
  }

  // Integer types fallback
  SmallVector<APInt> values;
  values.reserve(totalElements);
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  APInt zeroVal(bitWidth, 0);
  APInt oneVal(bitWidth, 1);

  if (shape.size() == 2) {
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        values.push_back((r == c) ? oneVal : zeroVal);
      }
    }
  } else if (shape.size() == 3) {
    int64_t batch = shape[0];
    int64_t rows = shape[1];
    int64_t cols = shape[2];
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          values.push_back((r == c) ? oneVal : zeroVal);
        }
      }
    }
  }
  auto attr = DenseElementsAttr::get(shapedType, values);
  return arith::ConstantOp::create(rewriter, loc, shapedType, attr);
}

Value createZeroFillTensor(PatternRewriter& rewriter, Location loc,
                           RankedTensorType type) {
  Type elemType = type.getElementType();
  Value empty =
      tensor::EmptyOp::create(rewriter, loc, type.getShape(), elemType);
  Value zero;
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    zero = arith::ConstantOp::create(
        rewriter, loc, elemType,
        rewriter.getFloatAttr(elemType,
                              APFloat(floatType.getFloatSemantics(), 0)));
  } else {
    zero = arith::ConstantOp::create(rewriter, loc, elemType,
                                     rewriter.getIntegerAttr(elemType, 0));
  }
  return linalg::FillOp::create(rewriter, loc, zero, empty).getResult(0);
}

struct PadToIdentityMatmulPattern : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp op,
                                PatternRewriter& rewriter) const override {
    auto sourceType = dyn_cast<RankedTensorType>(op.getSource().getType());
    auto targetType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!sourceType || !targetType || !sourceType.hasStaticShape() ||
        !targetType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires static tensor types");
    }

    // Verify low padding is all zeros
    for (int64_t low : op.getStaticLow()) {
      if (low != 0) {
        return rewriter.notifyMatchFailure(
            op, "non-zero low padding is not supported for identity matmul");
      }
    }

    // Verify constant zero fill inside pad body
    auto& block = op.getRegion().front();
    auto yieldOp = dyn_cast<tensor::YieldOp>(block.getTerminator());
    if (!yieldOp) {
      return rewriter.notifyMatchFailure(op,
                                         "missing tensor.yield in pad body");
    }
    Value padVal = yieldOp.getOperand();
    if (!matchPattern(padVal, m_AnyZeroFloat()) &&
        !matchPattern(padVal, m_Zero())) {
      return rewriter.notifyMatchFailure(op, "only zero padding is supported");
    }

    Location loc = op.getLoc();
    Type elemType = sourceType.getElementType();
    int64_t rank = sourceType.getRank();

    // 2D Tensor Padding [M, K] -> [M', K']
    if (rank == 2) {
      int64_t mIn = sourceType.getDimSize(0);
      int64_t kIn = sourceType.getDimSize(1);
      int64_t mOut = targetType.getDimSize(0);
      int64_t kOut = targetType.getDimSize(1);

      if (mIn == mOut && kIn == kOut) {
        rewriter.replaceOp(op, op.getSource());
        return success();
      }

      // Case 1: Row padding only (M -> M')
      if (mOut > mIn && kIn == kOut) {
        Value pl = createIdentityConstant(rewriter, loc, elemType, {mOut, mIn},
                                          mOut, mIn);
        Value init = createZeroFillTensor(rewriter, loc, targetType);
        auto matmul = linalg::MatmulOp::create(
            rewriter, loc, ValueRange{pl, op.getSource()}, ValueRange{init});
        rewriter.replaceOp(op, matmul.getResult(0));
        return success();
      }

      // Case 2: Column padding only (K -> K')
      if (mIn == mOut && kOut > kIn) {
        Value pr = createIdentityConstant(rewriter, loc, elemType, {kIn, kOut},
                                          kIn, kOut);
        Value init = createZeroFillTensor(rewriter, loc, targetType);
        auto matmul = linalg::MatmulOp::create(
            rewriter, loc, ValueRange{op.getSource(), pr}, ValueRange{init});
        rewriter.replaceOp(op, matmul.getResult(0));
        return success();
      }

      // Case 3: 2D Simultaneous padding (M -> M', K -> K')
      if (mOut > mIn && kOut > kIn) {
        auto midType = RankedTensorType::get({mOut, kIn}, elemType);
        Value pl = createIdentityConstant(rewriter, loc, elemType, {mOut, mIn},
                                          mOut, mIn);
        Value midInit = createZeroFillTensor(rewriter, loc, midType);
        auto rowMatmul = linalg::MatmulOp::create(
            rewriter, loc, ValueRange{pl, op.getSource()}, ValueRange{midInit});

        Value pr = createIdentityConstant(rewriter, loc, elemType, {kIn, kOut},
                                          kIn, kOut);
        Value finalInit = createZeroFillTensor(rewriter, loc, targetType);
        auto colMatmul = linalg::MatmulOp::create(
            rewriter, loc, ValueRange{rowMatmul.getResult(0), pr},
            ValueRange{finalInit});
        rewriter.replaceOp(op, colMatmul.getResult(0));
        return success();
      }
    }

    // 3D Batched Tensor Padding [H, M, K] -> [H, M', K']
    if (rank == 3) {
      int64_t hIn = sourceType.getDimSize(0);
      int64_t mIn = sourceType.getDimSize(1);
      int64_t kIn = sourceType.getDimSize(2);
      int64_t hOut = targetType.getDimSize(0);
      int64_t mOut = targetType.getDimSize(1);
      int64_t kOut = targetType.getDimSize(2);

      if (hIn != hOut) {
        return rewriter.notifyMatchFailure(
            op, "batch dimension padding is not supported");
      }

      if (mIn == mOut && kIn == kOut) {
        rewriter.replaceOp(op, op.getSource());
        return success();
      }

      // Case 1: Batch Row padding (M -> M')
      if (mOut > mIn && kIn == kOut) {
        Value pl = createIdentityConstant(rewriter, loc, elemType,
                                          {hIn, mOut, mIn}, mOut, mIn);
        Value init = createZeroFillTensor(rewriter, loc, targetType);
        auto bmm = linalg::BatchMatmulOp::create(
            rewriter, loc, ValueRange{pl, op.getSource()}, ValueRange{init});
        bmm->setAttr("mgmt.force_bootstrap", rewriter.getUnitAttr());
        rewriter.replaceOp(op, bmm.getResult(0));
        return success();
      }

      // Case 2: Batch Column padding (K -> K')
      if (mIn == mOut && kOut > kIn) {
        Value pr = createIdentityConstant(rewriter, loc, elemType,
                                          {hIn, kIn, kOut}, kIn, kOut);
        Value init = createZeroFillTensor(rewriter, loc, targetType);
        auto bmm = linalg::BatchMatmulOp::create(
            rewriter, loc, ValueRange{op.getSource(), pr}, ValueRange{init});
        rewriter.replaceOp(op, bmm.getResult(0));
        return success();
      }

      // Case 3: 2D Batch Simultaneous padding (M -> M', K -> K')
      if (mOut > mIn && kOut > kIn) {
        auto midType = RankedTensorType::get({hIn, mOut, kIn}, elemType);
        Value pl = createIdentityConstant(rewriter, loc, elemType,
                                          {hIn, mOut, mIn}, mOut, mIn);
        Value midInit = createZeroFillTensor(rewriter, loc, midType);
        auto rowBmm = linalg::BatchMatmulOp::create(
            rewriter, loc, ValueRange{pl, op.getSource()}, ValueRange{midInit});

        Value pr = createIdentityConstant(rewriter, loc, elemType,
                                          {hIn, kIn, kOut}, kIn, kOut);
        Value finalInit = createZeroFillTensor(rewriter, loc, targetType);
        auto colBmm = linalg::BatchMatmulOp::create(
            rewriter, loc, ValueRange{rowBmm.getResult(0), pr},
            ValueRange{finalInit});
        rewriter.replaceOp(op, colBmm.getResult(0));
        return success();
      }
    }

    return rewriter.notifyMatchFailure(
        op, "tensor rank not supported for identity matmul lowering");
  }
};

struct PushPadThroughMatmulPattern : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp op,
                                PatternRewriter& rewriter) const override {
    auto bmm = op.getSource().getDefiningOp<linalg::BatchMatmulOp>();
    if (!bmm) return failure();

    auto sourceType = dyn_cast<RankedTensorType>(op.getSource().getType());
    auto targetType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!sourceType || !targetType || !sourceType.hasStaticShape() ||
        !targetType.hasStaticShape() || sourceType.getRank() != 3) {
      return failure();
    }

    for (int64_t low : op.getStaticLow()) {
      if (low != 0) return failure();
    }

    int64_t mIn = sourceType.getDimSize(1);
    int64_t kIn = sourceType.getDimSize(2);
    int64_t mOut = targetType.getDimSize(1);
    int64_t kOut = targetType.getDimSize(2);

    // Only rewrite pure row padding (M -> M', K unchanged)
    if (mOut <= mIn || kIn != kOut) return failure();

    Value lhs = bmm.getInputs()[0];
    Value rhs = bmm.getInputs()[1];
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    if (!lhsType || !lhsType.hasStaticShape() || lhsType.getRank() != 3) {
      return failure();
    }

    Location loc = op.getLoc();
    Type elemType = lhsType.getElementType();
    int64_t rowPad = mOut - mIn;
    SmallVector<OpFoldResult> lowOfr = {rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> highOfr = {rewriter.getIndexAttr(0),
                                         rewriter.getIndexAttr(rowPad),
                                         rewriter.getIndexAttr(0)};
    auto paddedLhsType = RankedTensorType::get(
        {lhsType.getDimSize(0), mOut, lhsType.getDimSize(2)}, elemType);

    Value zero;
    if (auto floatType = dyn_cast<FloatType>(elemType)) {
      zero = arith::ConstantOp::create(
          rewriter, loc, elemType,
          rewriter.getFloatAttr(elemType,
                                APFloat(floatType.getFloatSemantics(), 0)));
    } else {
      zero = arith::ConstantOp::create(rewriter, loc, elemType,
                                       rewriter.getIntegerAttr(elemType, 0));
    }

    auto newPad = rewriter.create<tensor::PadOp>(loc, paddedLhsType, lhs,
                                                 lowOfr, highOfr);
    Region& region = newPad.getRegion();
    Block* block = rewriter.createBlock(&region);
    rewriter.setInsertionPointToStart(block);
    rewriter.create<tensor::YieldOp>(loc, zero);
    rewriter.setInsertionPointAfter(newPad);

    Value newInit = createZeroFillTensor(rewriter, loc, targetType);
    auto newBmm = rewriter.create<linalg::BatchMatmulOp>(
        loc, targetType, ValueRange{newPad.getResult(), rhs},
        ValueRange{newInit});

    rewriter.replaceOp(op, newBmm.getResult(0));
    return success();
  }
};

struct LowerPadToIdentityMatmul
    : public impl::LowerPadToIdentityMatmulBase<LowerPadToIdentityMatmul> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    // Directly lower tensor.pad on intermediate tensors to Left PCMM
    // without pushing pad through preceding matmul.
    patterns.add<PadToIdentityMatmulPattern>(context, 1);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
