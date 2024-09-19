#include "lib/Dialect/TensorExt/Transforms/RotateAndReduce.h"

#include <cstdint>

#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/DenseSet.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Iterators.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project

#define DEBUG_NAME "rotate-and-reduce"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_ROTATEANDREDUCE
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

/// A pass that searches for a length N sequence of binary operations that
/// reduces a length N vector to a single scalar, and replaces it with a
/// logarithmic number of rotations and binary operations.
struct RotateAndReduce : impl::RotateAndReduceBase<RotateAndReduce> {
  using RotateAndReduceBase::RotateAndReduceBase;

  template <typename ArithOp>
  void tryReplaceRotations(ArithOp op,
                           const rotation_analysis::PartialReduction &reduction,
                           bool extraction) {
    LLVM_DEBUG(llvm::dbgs()
               << "Trying to replace rotations ending in " << *op << "\n");
    auto b = ImplicitLocOpBuilder(op->getLoc(), op);
    auto tensor = reduction.getTensor();
    Operation *finalOp;
    auto tensorShape =
        mlir::cast<RankedTensorType>(tensor.getType()).getShape();
    for (int64_t shiftSize = tensorShape[0] / 2; shiftSize > 0;
         shiftSize /= 2) {
      auto rotatedTensor = b.create<tensor_ext::RotateOp>(
          tensor, b.create<arith::ConstantOp>(b.getIndexAttr(shiftSize)));
      auto addOp = b.create<ArithOp>(tensor, rotatedTensor);
      finalOp = addOp;
      tensor = addOp->getResult(0);
    }

    [[maybe_unused]] auto *parentOp = op->getParentOp();
    if (extraction) {
      // We can extract at any index; every index contains the same reduced
      // value.
      auto extractOp = b.create<tensor::ExtractOp>(
          finalOp->getResult(0),
          b.create<arith::ConstantIndexOp>(0).getResult());
      op->replaceAllUsesWith(extractOp);
    } else {
      op->replaceAllUsesWith(finalOp);
    }
    LLVM_DEBUG(llvm::dbgs() << "Post-replacement: " << *parentOp << "\n");
  }

  void runOnOperation() override {
    DataFlowSolver solver;
    // These two upstream analyses are required dependencies for any sparse
    // dataflow analysis, or else the analysis will be a no-op. Cf.
    // https://github.com/llvm/llvm-project/issues/58922
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run dataflow analysis.\n";
      signalPassFailure();
      return;
    }

    rotation_analysis::RotationAnalysis rotationAnalysis(solver);
    rotationAnalysis.run(getOperation());

    getOperation()->walk<WalkOrder::PreOrder, ReverseIterator>(
        [&](Operation *op) {
          for (Value result : op->getResults()) {
            if (!rotationAnalysis.containsRootedReductions(result)) {
              continue;
            }

            // When the reduction ends with a scalar addi/muli, that means
            // multiple extraction from tensor has been done earlier, and
            // can be optimized to be only one final extraction
            bool extraction = !mlir::isa<RankedTensorType>(result.getType());

            for (const auto &reduction :
                 rotationAnalysis.getRootedReductionsAt(result)) {
              if (reduction.isComplete()) {
                llvm::TypeSwitch<Operation &>(*op)
                    .Case<arith::AddIOp>([&](auto arithOp) {
                      tryReplaceRotations<arith::AddIOp>(arithOp, reduction,
                                                         extraction);
                    })
                    .Case<arith::MulIOp>([&](auto arithOp) {
                      tryReplaceRotations<arith::MulIOp>(arithOp, reduction,
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
