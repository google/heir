#include "lib/Dialect/TfheRust/IR/TfheRustPatterns.h"

#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "mlir/include/mlir/IR/Block.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Dominance.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tfhe_rust {

// Move an operation as early as possible, so long as it only depends on the
// server key. This pattern has the downside of adding an annotation to the
// moved op, but this is necsesary to avoid infinite loops in the pattern
// matcher. A later pass could remove the attributes, but they are harmless
// and not emitted in the final codegen.
template <typename SksOp>
LogicalResult doHoist(SksOp op, PatternRewriter& rewriter) {
  if (op->hasAttr("hoisted")) {
    return failure();
  }
  DominanceInfo dom(op);
  Operation* lastOperandDefiner = nullptr;
  Block* lastBlock = nullptr;
  for (Value operand : op->getOperands()) {
    if (auto* defOp = operand.getDefiningOp()) {
      if (lastOperandDefiner == nullptr ||
          dom.dominates(lastOperandDefiner, defOp)) {
        lastOperandDefiner = defOp;
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      Block* block = blockArg.getOwner();
      if (lastBlock == nullptr || dom.dominates(lastBlock, block)) {
        lastBlock = block;
      }
    }
  }

  // The last argument's defining thing is a block
  if (lastOperandDefiner == nullptr ||
      dom.dominates(lastOperandDefiner, lastBlock->getParentOp())) {
    rewriter.moveOpBefore(op, &lastBlock->getOperations().front());
    op->setAttr("hoisted", rewriter.getBoolAttr(true));
    return success();
  }

  rewriter.moveOpAfter(op, lastOperandDefiner);
  op->setAttr("hoisted", rewriter.getBoolAttr(true));
  return success();
}

LogicalResult HoistGenerateLookupTable::matchAndRewrite(
    GenerateLookupTableOp op, PatternRewriter& rewriter) const {
  return doHoist(op, rewriter);
}

LogicalResult HoistCreateTrivial::matchAndRewrite(
    CreateTrivialOp op, PatternRewriter& rewriter) const {
  return doHoist(op, rewriter);
}

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
