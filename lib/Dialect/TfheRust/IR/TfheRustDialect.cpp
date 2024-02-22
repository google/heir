#include "include/Dialect/TfheRust/IR/TfheRustDialect.h"

#include "include/Dialect/TfheRust/IR/TfheRustDialect.cpp.inc"
#include "include/Dialect/TfheRust/IR/TfheRustOps.h"
#include "include/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dominance.h"              // from @llvm-project

#define GET_TYPEDEF_CLASSES
#include "include/Dialect/TfheRust/IR/TfheRustTypes.cpp.inc"
#define GET_OP_CLASSES
#include "include/Dialect/TfheRust/IR/TfheRustOps.cpp.inc"

namespace mlir {
namespace heir {
namespace tfhe_rust {

void TfheRustDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "include/Dialect/TfheRust/IR/TfheRustTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/TfheRust/IR/TfheRustOps.cpp.inc"
      >();
}

// Move an operation as early as possible, so long as it only depends on the
// server key. This pattern has the downside of adding an annotation to the
// moved op, but this is necsesary to avoid infinite loops in the pattern
// matcher. A later pass could remove the attributes, but they are harmless
// and not emitted in the final codegen.
template <typename SksOp>
LogicalResult HoistConstantLikeOps<SksOp>::matchAndRewrite(
    SksOp op, PatternRewriter &rewriter) const {
  if (op->hasAttr("hoisted")) {
    return failure();
  }
  DominanceInfo dom(op);
  Operation *lastOperandDefiner = nullptr;
  Block *lastBlock = nullptr;
  for (Value operand : op->getOperands()) {
    if (auto *defOp = operand.getDefiningOp()) {
      if (lastOperandDefiner == nullptr ||
          dom.dominates(lastOperandDefiner, defOp)) {
        lastOperandDefiner = defOp;
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      Block *block = blockArg.getOwner();
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

void GenerateLookupTableOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<HoistConstantLikeOps<GenerateLookupTableOp>>(context);
}

void CreateTrivialOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<HoistConstantLikeOps<CreateTrivialOp>>(context);
}

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
