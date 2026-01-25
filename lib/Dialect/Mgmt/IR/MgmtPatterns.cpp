#include "lib/Dialect/Mgmt/IR/MgmtPatterns.h"

#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mgmt {

LogicalResult ReplaceWithLevelReduce::matchAndRewrite(
    LevelReduceMinOp op, mlir::PatternRewriter& rewriter) const {
  auto resultMgmtAttr = findMgmtAttrAssociatedWith(op.getResult());
  auto operandMgmtAttr = findMgmtAttrAssociatedWith(op.getInput());
  if (!resultMgmtAttr || !operandMgmtAttr) {
    return failure();
  }

  if (resultMgmtAttr.getLevel() == operandMgmtAttr.getLevel()) {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }

  // This is an invalid state: the op reduces level by definition
  if (resultMgmtAttr.getLevel() > operandMgmtAttr.getLevel()) {
    return failure();
  }

  int64_t levelDiff = operandMgmtAttr.getLevel() - resultMgmtAttr.getLevel();
  rewriter.replaceOpWithNewOp<LevelReduceOp>(
      op, op.getInput(), rewriter.getI64IntegerAttr(levelDiff));

  return success();
}

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
