#include "lib/Transforms/PopulateScale/PopulateScalePatterns.h"

#include <cstdint>

#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {

template <typename MulOp>
LogicalResult ConvertAdjustScaleToMulPlain<MulOp>::matchAndRewrite(
    mgmt::AdjustScaleOp op, PatternRewriter& rewriter) const {
  auto inputScale = mgmt::findMgmtAttrAssociatedWith(op.getInput()).getScale();
  int64_t scale = mgmt::findMgmtAttrAssociatedWith(op).getScale();
  // no need to adjust scale
  if (scale == inputScale) {
    rewriter.replaceAllOpUsesWith(op, op->getOperand(0));
    rewriter.eraseOp(op);
    return success();
  }

  auto deltaScale = materializer->deltaScale(scale, inputScale);
  if (deltaScale < 0) {
    op.emitError() << "delta scale is negative";
    return failure();
  }

  // lower to (input * all_ones)

  auto mgmtAttr =
      op->getAttrOfType<mgmt::MgmtAttr>(mgmt::MgmtDialect::kArgMgmtAttrName);

  auto inputType = op.getInput().getType();
  auto oneAttr = rewriter.getOneAttr(inputType);
  if (!oneAttr) {
    return op.emitOpError() << "Unsupported type for lowering";
  }

  // create arith.constant at the beginning of the function
  auto funcOp = op->getParentOfType<func::FuncOp>();
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  auto allOnes = mlir::arith::ConstantOp::create(rewriter, op.getLoc(),
                                                 inputType, oneAttr);

  // insert following ops right before op
  rewriter.setInsertionPoint(op);
  // do not annotate deltaScale to arith.constant as canonicalizer will merge
  // arith.constant with same value and mgmt attr will be lost

  // init-op also for preventing mul 1 being constant folded
  auto initOp = mgmt::InitOp::create(rewriter, op.getLoc(), inputType,
                                     allOnes.getResult());
  mgmt::setMgmtAttrAssociatedWith(
      initOp, getMgmtAttrWithNewScale(mgmtAttr, deltaScale));
  auto mulOp = MulOp::create(rewriter, op.getLoc(), inputType, op.getInput(),
                             initOp.getOutput());
  mgmt::setMgmtAttrAssociatedWith(mulOp, mgmtAttr);
  rewriter.replaceAllOpUsesWith(op, mulOp.getResult());
  return success();
}

// instantiation
template struct ConvertAdjustScaleToMulPlain<arith::MulIOp>;
template struct ConvertAdjustScaleToMulPlain<arith::MulFOp>;

}  // namespace heir
}  // namespace mlir
