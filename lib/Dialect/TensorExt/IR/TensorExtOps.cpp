#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

// Kept inside a namespace because it generates a function called
// populateWithGenerated, which can conflict with other generated patterns.
#include "lib/Dialect/TensorExt/IR/TensorExtCanonicalization.cpp.inc"

void RotateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  populateWithGenerated(results);
}

LogicalResult RotateOp::verify() {
  // If 2-D tensor, then expect 2 dimensions to rotate.
  auto x = getTensor().getType();
  if (x.getRank() != 1) {
    bool foundNonOne = false;
    for (auto dim : x.getShape()) {
      if (dim != 1) {
        if (foundNonOne) {
          return emitOpError()
                 << "requires a 1-D input tensor, but found " << x;
        }
        foundNonOne = true;
      }
    }
  }
  return success();
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
