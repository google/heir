#include "lib/Dialect/TensorExt/Transforms/Patterns.h"

#include <cstdint>

#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

LogicalResult FoldConvertLayoutIntoAssignLayoutPattern::matchAndRewrite(
    AssignLayoutOp op, PatternRewriter& rewriter) const {
  if (op.getResult().getUsers().empty()) {
    rewriter.eraseOp(op);
    return success();
  }

  int64_t numConverted = 0;
  // Can't modify the users while iterating over them, so copy them to a
  // vector first.
  SmallVector<OpOperand, 4> users(op.getResult().getUsers());
  for (const OpOperand& user : users) {
    Operation* owner = user.getOwner();
    if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(owner)) {
      if (convertLayoutOp.getFromLayout() != op.getLayout()) {
        // This should be considered invalid, but check again here for
        // safety.
        continue;
      }

      DenseI64ArrayAttr domainSchedule =
          !convertLayoutOp.getDomainSchedule().empty()
              ? convertLayoutOp.getDomainScheduleAttr()
              : op.getDomainScheduleAttr();

      auto newOp = AssignLayoutOp::create(
          rewriter, convertLayoutOp.getLoc(), op.getValue(),
          convertLayoutOp.getToLayout(), domainSchedule);
      rewriter.replaceOp(convertLayoutOp, newOp.getResult());
      // Ensure the newOp has its layout attribute properly set
      setAttributeAssociatedWith(newOp.getResult(),
                                 TensorExtDialect::kLayoutAttrName,
                                 newOp.getLayout());
      ++numConverted;
    }
  }

  return numConverted > 0 ? success() : failure();
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
