#ifndef LIB_DIALECT_MGMT_IR_MGMTPATTERNS_H_
#define LIB_DIALECT_MGMT_IR_MGMTPATTERNS_H_

#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mgmt {

struct ReplaceWithLevelReduce
    : public mlir::OpRewritePattern<LevelReduceMinOp> {
 public:
  using OpRewritePattern<LevelReduceMinOp>::OpRewritePattern;

  llvm::LogicalResult matchAndRewrite(
      LevelReduceMinOp op, mlir::PatternRewriter& rewriter) const override;
};

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MGMT_IR_MGMTPATTERNS_H_
