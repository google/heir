#include "lib/Transforms/OperationBalancer/OperationBalancer.h"

#include <set>
#include <stack>
#include <vector>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

#define DEBUG_TYPE "operation-balancer"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_OPERATIONBALANCER
#include "lib/Transforms/OperationBalancer/OperationBalancer.h.inc"

template <typename OpType>
OpType recursiveProduceBalancedTree(OpBuilder &builder, Location &loc,
                                    std::vector<Value> &flattenedOperands) {
  // If there are only two operands, return the operation over the two operands.
  // Else if there are three operands, produce an operation with two of the
  // operands, and produce another operation with the third operand.
  // Else, split the operands in half and recursively call this function on each
  // half, creating an appropriate operation for each half.
  if (flattenedOperands.size() == 2) {
    return builder.create<OpType>(loc, flattenedOperands[0],
                                  flattenedOperands[1]);
  } else if (flattenedOperands.size() == 3) {
    std::vector<Value> leftOperands;
    leftOperands.reserve(2);
    leftOperands.push_back(flattenedOperands[0]);
    leftOperands.push_back(flattenedOperands[1]);

    auto leftTree =
        recursiveProduceBalancedTree<OpType>(builder, loc, leftOperands);
    return builder.create<OpType>(loc, leftTree, flattenedOperands.back());
  } else {
    // split the operands in half
    int leftSize = flattenedOperands.size() / 2;

    std::vector<Value> leftOperands;
    leftOperands.reserve(leftSize);
    std::vector<Value> rightOperands;
    rightOperands.reserve(flattenedOperands.size() - leftSize);

    for (size_t i = 0; i < flattenedOperands.size(); i++) {
      if (i < (size_t)leftSize) {
        leftOperands.push_back(flattenedOperands[i]);
      } else {
        rightOperands.push_back(flattenedOperands[i]);
      }
    }

    auto leftTree =
        recursiveProduceBalancedTree<OpType>(builder, loc, leftOperands);
    auto rightTree =
        recursiveProduceBalancedTree<OpType>(builder, loc, rightOperands);

    // create an appropriate operation for the two halves
    return builder.create<OpType>(loc, leftTree, rightTree);
  }
}

template <typename OpType>
void tryBalanceBlock(Block *block) {
  // visited set checks whether we've already handled an operation
  std::set<Operation *> visited;

  std::vector<std::vector<Operation *>> deleteOpsOrderLists;
  std::vector<std::vector<Value>> operandsLists;
  std::vector<Operation *> roots;

  // Balance from backwards to front.
  for (auto &op : llvm::reverse(block->getOperations())) {
    if ((!llvm::isa<OpType>(&op)) || (visited.find(&op) != visited.end())) {
      continue;
    }

    // deleteOpsOrder will be used to erase the operations in a reverse
    // topological ordering.
    std::vector<Operation *> deleteOpsOrder;

    // These operands are the flattened operands in a tree of operations.
    std::vector<Value> operands;

    // This stack is used to traverse the tree of operations in a depth-first
    // search. Depth-first search enables us to find operands from the left to
    // the right of the tree in order. This ultimately may not matter because
    // the operands may be sorted anyway (and additions and multiplications are
    // commutative).
    std::stack<Value> stack;

    // Since this is the first operation we are checking, (and this is an
    // operation we are trying to balance), then we will check its children to
    // find operands because we know that this is not an operand.
    visited.insert(&op);
    deleteOpsOrder.push_back(&op);
    stack.push(op.getOperand(1));
    stack.push(op.getOperand(0));

    while (!stack.empty()) {
      Value current = stack.top();
      Operation *currentOp = current.getDefiningOp();
      stack.pop();

      // The following condition checks whether the current operation is an
      // operand or not.
      // It is not an operand if it is an operation we are trying to balance and
      // also has only one use, so we will check its children to find more
      // operands.
      if (currentOp != nullptr && llvm::isa<OpType>(currentOp) &&
          currentOp->hasOneUse()) {
        // Not an operand, but instead a temporary value in a larger tree of
        // operations we are trying to balance, so we will continue to its
        // children.
        visited.insert(currentOp);
        deleteOpsOrder.push_back(currentOp);
        stack.push(currentOp->getOperand(1));
        stack.push(currentOp->getOperand(0));
      } else {
        // This is an operand because it either has multiple uses or is not an
        // operation we are trying to balance.
        // If it has multiple uses, then it is a useful intermediate value we
        // want to preserve. Right now, we are not trying to optimally reduce
        // the depth by adding extra operations because of these intermediate
        // values. Instead, we are making a cut in the tree and balancing each
        // individually.
        operands.push_back(current);

        // TODO (#836): an implicit assumption here is that the intermediate
        // values are treated has having the same depth as other operands, so
        // our balancing doesn't account for that optimally when producing a
        // balanced tree. What probably needs to change is to balance
        // intermediate values first, keep track of the depth of the
        // intermediate values, and use that to balance a tree that uses this
        // intermediate value as an operand.
      }
    }

    deleteOpsOrderLists.push_back(deleteOpsOrder);
    operandsLists.push_back(operands);
    roots.push_back(&op);
  }

  for (size_t i = 0; i < roots.size(); i++) {
    std::vector<Operation *> deleteOpsOrder = deleteOpsOrderLists[i];
    std::vector<Value> operands = operandsLists[i];
    Operation *root = roots[i];

    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      llvm::dbgs() << "Here are the flattened operands:" << "\n";
      for (auto value : operands) {
        llvm::dbgs() << "Value: " << value << "\n";
      }
      llvm::dbgs() << "\n";
    });

    // If there are fewer than 3 operands, there's nothing to balance.
    if (operands.size() <= 2) {
      continue;
    }

    // Now that we have the flattened operands, we can create the balanced tree
    // of operations.
    Location loc = root->getLoc();
    OpBuilder builder(root);
    OpType newOps =
        recursiveProduceBalancedTree<OpType>(builder, loc, operands);
    root->replaceAllUsesWith(newOps);

    // Delete the old operations.
    for (auto op : deleteOpsOrder) {
      op->erase();
    }
  }
}

struct OperationBalancer : impl::OperationBalancerBase<OperationBalancer> {
  using OperationBalancerBase::OperationBalancerBase;

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
      // TODO (#836): analyze the block to see which operations should be done
      // secretly (i.e. under homomorphic encryption) to determine which
      // operations are done plaintext and others over ciphertext. This is
      // useful because we can then sort the operations in a way that minimizes
      // the number of encodings.
      tryBalanceBlock<arith::AddIOp>(genericOp.getBody());
      tryBalanceBlock<arith::MulIOp>(genericOp.getBody());
      tryBalanceBlock<arith::AddFOp>(genericOp.getBody());
      tryBalanceBlock<arith::MulFOp>(genericOp.getBody());
    });
  }
};

}  // namespace heir
}  // namespace mlir
