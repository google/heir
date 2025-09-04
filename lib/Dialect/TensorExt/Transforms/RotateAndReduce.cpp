#include "lib/Dialect/TensorExt/Transforms/RotateAndReduce.h"

#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/TensorExt/Transforms/ImplementRotateAndReduce.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Iterators.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project

#define DEBUG_TYPE "rotate-and-reduce"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_ROTATEANDREDUCE
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

/// A pass that searches for a length N sequence of add operations that
/// reduces a length N vector to a single scalar, and replaces it with a
/// logarithmic number of rotations and binary operations.
struct RotateAndReduce : impl::RotateAndReduceBase<RotateAndReduce> {
  using RotateAndReduceBase::RotateAndReduceBase;

  template <typename ArithOp>
  void tryReplaceRotations(ArithOp op,
                           const rotation_analysis::PartialReduction& reduction,
                           bool extraction) {
    LLVM_DEBUG(llvm::dbgs()
               << "Trying to replace rotations ending in " << *op << "\n");
    auto b = ImplicitLocOpBuilder(op->getLoc(), op);
    auto tensor = reduction.getTensor();
    auto tensorShape =
        mlir::cast<RankedTensorType>(tensor.getType()).getShape();

    // Get the operation name for the reduce_op attribute
    auto rotateAndReduceOp = tensor_ext::RotateAndReduceOp::create(
        b, tensor,
        /*period=*/1,
        /*steps=*/tensorShape[0],
        /*reduceOp=*/op->getName().getStringRef());
    Operation* finalOp = rotateAndReduceOp;

    [[maybe_unused]] auto* parentOp = op->getParentOp();
    if (extraction) {
      // We can extract at any index; every index contains the same reduced
      // value.
      auto extractOp = tensor::ExtractOp::create(
          b, finalOp->getResult(0),
          arith::ConstantIndexOp::create(b, 0).getResult());
      finalOp = extractOp;
    }
    for (auto value : reduction.getSavedValues()) {
      finalOp = ArithOp::create(b, finalOp->getResult(0), value);
    }
    if (finalOp) op->replaceAllUsesWith(finalOp);
    LLVM_DEBUG(llvm::dbgs() << "Post-replacement: " << *parentOp << "\n");

    // Convert the rotate_and_reduce op to its implementation immediately
    if (failed(convertRotateAndReduceOp(rotateAndReduceOp))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to convert rotate_and_reduce op\n");
      return;
    }
  }

  void runOnOperation() override {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run dataflow analysis.\n";
      signalPassFailure();
      return;
    }

    rotation_analysis::RotationAnalysis rotationAnalysis(solver);
    rotationAnalysis.run(getOperation());

    getOperation()->walk<WalkOrder::PreOrder, ReverseIterator>(
        [&](Operation* op) {
          for (Value result : op->getResults()) {
            if (!rotationAnalysis.containsRootedReductions(result)) {
              continue;
            }

            // When the reduction ends with a scalar addi/muli, that means
            // multiple extraction from tensor has been done earlier, and
            // can be optimized to be only one final extraction
            bool extraction = !mlir::isa<RankedTensorType>(result.getType());

            for (const auto& reduction :
                 rotationAnalysis.getRootedReductionsAt(result)) {
              if (reduction.isComplete() &&
                  cast<RankedTensorType>(reduction.getTensor().getType())
                          .getNumElements() > 1) {
                llvm::TypeSwitch<Operation&>(*op)
                    .Case<arith::AddIOp>([&](auto arithOp) {
                      tryReplaceRotations<arith::AddIOp>(arithOp, reduction,
                                                         extraction);
                    })
                    .Case<arith::AddFOp>([&](auto arithOp) {
                      tryReplaceRotations<arith::AddFOp>(arithOp, reduction,
                                                         extraction);
                    })
                    .Case<arith::MulIOp>([&](auto arithOp) {
                      tryReplaceRotations<arith::MulIOp>(arithOp, reduction,
                                                         extraction);
                    })
                    .Case<arith::MulFOp>([&](auto arithOp) {
                      tryReplaceRotations<arith::MulFOp>(arithOp, reduction,
                                                         extraction);
                    });
              }
            }
          }
        });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
