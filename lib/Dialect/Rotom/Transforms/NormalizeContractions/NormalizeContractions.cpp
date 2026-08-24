#include "lib/Dialect/Rotom/Transforms/NormalizeContractions/NormalizeContractions.h"

#include <cstdint>
#include <utility>

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::rotom {

#define GEN_PASS_DEF_NORMALIZECONTRACTIONS
#include "lib/Dialect/Rotom/Transforms/NormalizeContractions/NormalizeContractions.h.inc"

namespace {

// Whether `init` is a zero accumulator: a linalg.fill of zero or a zero
// splat constant.
bool isZeroFill(Value init) {
  if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
    return matchPattern(fill.getInputs()[0], m_AnyZeroFloat()) ||
           matchPattern(fill.getInputs()[0], m_Zero());
  }
  DenseElementsAttr splat;
  if (matchPattern(init, m_Constant(&splat))) {
    if (!splat.isSplat()) return false;
    if (auto f = dyn_cast<FloatAttr>(splat.getSplatValue<Attribute>())) {
      return f.getValue().isZero();
    }
    if (auto i = dyn_cast<IntegerAttr>(splat.getSplatValue<Attribute>())) {
      return i.getValue().isZero();
    }
  }
  return false;
}

// linalg.matvec A(MxK) * x(K) -> collapse(linalg.matmul A * expand(x, Kx1)).
struct MatvecToMatmul : OpRewritePattern<linalg::MatvecOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatvecOp op,
                                PatternRewriter& rewriter) const override {
    Value matrix = op.getInputs()[0];
    Value vec = op.getInputs()[1];
    Value init = op.getOutputs()[0];
    auto vecType = cast<RankedTensorType>(vec.getType());
    auto initType = cast<RankedTensorType>(init.getType());
    if (vecType.getRank() != 1 || initType.getRank() != 1 ||
        !vecType.hasStaticShape() || !initType.hasStaticShape()) {
      return failure();
    }
    Type elt = vecType.getElementType();
    auto colType = RankedTensorType::get({vecType.getDimSize(0), 1}, elt);
    auto resColType = RankedTensorType::get({initType.getDimSize(0), 1}, elt);

    Location loc = op.getLoc();
    SmallVector<ReassociationIndices> reassoc = {{0, 1}};
    Value colVec =
        tensor::ExpandShapeOp::create(rewriter, loc, colType, vec, reassoc);
    Value colInit =
        tensor::ExpandShapeOp::create(rewriter, loc, resColType, init, reassoc);
    auto matmul = linalg::MatmulOp::create(rewriter, loc, TypeRange{resColType},
                                           ValueRange{matrix, colVec},
                                           ValueRange{colInit});
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        op, matmul.getResult(0), reassoc);
    return success();
  }
};

// A matmul accumulating into a non-zero init (e.g. a bias folded into
// `outs` by linalg fusion) becomes a zero-filled matmul plus an explicit
// elementwise add of the init.
struct SplitNonZeroInit : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter& rewriter) const override {
    Value init = op.getOutputs()[0];
    if (isZeroFill(init)) return failure();
    auto resType = cast<RankedTensorType>(op.getResult(0).getType());
    if (!resType.hasStaticShape()) return failure();
    Type elt = resType.getElementType();
    if (!isa<FloatType>(elt) && !isa<IntegerType>(elt)) return failure();

    Location loc = op.getLoc();
    Value zero = arith::ConstantOp::create(rewriter, loc, elt,
                                           rewriter.getZeroAttr(elt));
    Value empty =
        tensor::EmptyOp::create(rewriter, loc, resType.getShape(), elt);
    Value fill =
        linalg::FillOp::create(rewriter, loc, zero, empty).getResult(0);
    auto matmul = linalg::MatmulOp::create(rewriter, loc, TypeRange{resType},
                                           op.getInputs(), ValueRange{fill});
    if (isa<FloatType>(elt)) {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, matmul.getResult(0), init);
    } else {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, matmul.getResult(0), init);
    }
    return success();
  }
};

}  // namespace

struct NormalizeContractions
    : public impl::NormalizeContractionsBase<NormalizeContractions> {
  using NormalizeContractionsBase::NormalizeContractionsBase;

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<MatvecToMatmul, SplitNonZeroInit>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::rotom
