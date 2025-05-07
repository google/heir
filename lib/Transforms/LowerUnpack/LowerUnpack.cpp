#include "lib/Transforms/LowerUnpack/LowerUnpack.h"

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project

// iwyu pragma: begin_keep
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project
// iwyu pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LOWERUNPACK
#include "lib/Transforms/LowerUnpack/LowerUnpack.h.inc"

struct LowerUnpack : impl::LowerUnpackBase<LowerUnpack> {
  using LowerUnpackBase::LowerUnpackBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter builder(context);
    ImplicitLocOpBuilder b(getOperation()->getLoc(), builder);
    auto result =
        getOperation()->walk<WalkOrder::PreOrder>([&](tensor_ext::UnpackOp op) {
          auto res = implementUnpackOp(op, b, [&](Operation *createdOp) {});
          if (failed(res)) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace heir
}  // namespace mlir
