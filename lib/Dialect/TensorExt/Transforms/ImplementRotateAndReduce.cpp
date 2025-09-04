#include "lib/Dialect/TensorExt/Transforms/ImplementRotateAndReduce.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Kernel/KernelImplementation.h"
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

#define DEBUG_TYPE "implement-rotate-and-reduce"

namespace mlir {
namespace heir {
namespace tensor_ext {

using ::mlir::heir::kernel::ArithmeticDagNode;
using ::mlir::heir::kernel::implementRotateAndReduce;
using ::mlir::heir::kernel::IRMaterializingVisitor;
using ::mlir::heir::kernel::SSAValue;

#define GEN_PASS_DEF_IMPLEMENTROTATEANDREDUCE
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

LogicalResult convertRotateAndReduceOp(RotateAndReduceOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Converting tensor_ext.rotate_and_reduce op: "
                          << op << "\n");
  TypedValue<RankedTensorType> input = op.getTensor();
  unsigned steps = op.getSteps().getZExtValue();
  unsigned period = op.getPeriod().getZExtValue();
  std::shared_ptr<ArithmeticDagNode<SSAValue>> implementedKernel;
  SSAValue vectorLeaf(input);
  std::optional<SSAValue> plaintextsLeaf = std::nullopt;

  if (op.getPlaintexts()) {
    plaintextsLeaf = std::optional<SSAValue>(op.getPlaintexts());
  }

  std::string reduceOp = "arith.addi";
  if (op.getReduceOp().has_value() && *op.getReduceOp() != nullptr) {
    reduceOp = op.getReduceOp()->getValue().str();
  }
  implementedKernel = implementRotateAndReduce(vectorLeaf, plaintextsLeaf,
                                               period, steps, reduceOp);
  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPointAfter(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  IRMaterializingVisitor visitor(b, input.getType());
  Value finalOutput = implementedKernel->visit(visitor);
  rewriter.replaceOp(op, finalOutput);
  return success();
}

struct ImplementRotateAndReduce
    : impl::ImplementRotateAndReduceBase<ImplementRotateAndReduce> {
  using ImplementRotateAndReduceBase::ImplementRotateAndReduceBase;

  void runOnOperation() override {
    getOperation()->walk([&](RotateAndReduceOp op) {
      if (failed(convertRotateAndReduceOp(op))) {
        op->emitOpError() << "failed to lower rotate_and_reduce op";
        signalPassFailure();
      }
    });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
