#include "lib/Dialect/CGGI/Transforms/BooleanLineVectorizer.h"

#include <string>

#include "lib/Dialect/CGGI/IR/CGGIAttributes.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Graph/Graph.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/TopologicalSortUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

#define DEBUG_TYPE "bool-line-vectorizer"

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_DEF_BOOLEANLINEVECTORIZER
#include "lib/Dialect/CGGI/Transforms/Passes.h.inc"

bool areCompatibleBool(Operation *lhs, Operation *rhs) {
  if (lhs->getDialect() != rhs->getDialect() ||
      lhs->getResultTypes() != rhs->getResultTypes() ||
      lhs->getAttrs() != rhs->getAttrs()) {
    return false;
  }
  // TODO: Check if can be made better with a BooleanPackableGate trait
  // on the CGGI_BinaryGateOp's?
  return OpTrait::hasElementwiseMappableTraits(lhs);
}

bool tryBoolVectorizeBlock(Block *block, MLIRContext &context) {
  graph::Graph<Operation *> graph;
  for (auto &op : block->getOperations()) {
    if (!op.hasTrait<OpTrait::Elementwise>()) {
      continue;
    }

    graph.addVertex(&op);
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions options;
    options.omitBlockArguments = true;

    getBackwardSlice(&op, &backwardSlice, options);
    for (auto *upstreamDep : backwardSlice) {
      // An edge from upstreamDep to `op` means that upstreamDep must be
      // computed before `op`.
      graph.addEdge(upstreamDep, &op);
    }
  }

  if (graph.empty()) {
    return false;
  }

  auto result = graph.sortGraphByLevels();
  assert(succeeded(result) &&
         "Only possible failure is a cycle in the SSA graph!");
  auto levels = result.value();

  LLVM_DEBUG({
    llvm::dbgs()
        << "Found operations to vectorize. In topo-sorted level order:\n";
    int level_num = 0;
    for (const auto &level : levels) {
      llvm::dbgs() << "\nLevel " << level_num++ << ":\n";
      for (auto op : level) {
        llvm::dbgs() << " - " << *op << "\n";
      }
    }
  });

  bool madeReplacement = false;
  for (const auto &level : levels) {
    DenseMap<Operation *, SmallVector<Operation *, 4>> compatibleOps;
    for (auto *op : level) {
      bool foundCompatible = false;
      for (auto &[key, bucket] : compatibleOps) {
        if (areCompatibleBool(key, op)) {
          compatibleOps[key].push_back(op);
          foundCompatible = true;
        }
      }
      if (!foundCompatible) {
        compatibleOps[op].push_back(op);
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Partitioned level of size " << level.size() << " into "
               << compatibleOps.size() << " groups of compatible ops\n");

    // Loop over all the compatibleOp groups
    // Each loop will have the key and a bucket with all the operations in
    for (auto &[key, bucket] : compatibleOps) {
      if (bucket.size() < 2) {
        continue;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[START] Bucket \t Vectorizing ops:\n";
        for (auto op : bucket) {
          llvm::dbgs() << " - " << *op << "\n";
        }
      });

      OpBuilder builder(bucket.back());
      // relies on CGGI ops having a single result type
      Type elementType = key->getResultTypes()[0];
      RankedTensorType tensorType = RankedTensorType::get(
          {static_cast<int64_t>(bucket.size())}, elementType);

      SmallVector<Value, 4> vectorizedOperands;
      SmallVector<StringAttr, 4> vectorizedGateOperands;

      for (auto *op : bucket) {
        std::string str;
        if (isa<cggi::AndOp>(op)) {
          str = "and";
        } else if (isa<cggi::NandOp>(op)) {
          str = "nand";
        } else if (isa<cggi::XorOp>(op)) {
          str = "xor";
        } else if (isa<cggi::XNorOp>(op)) {
          str = "xnor";
        } else if (isa<cggi::OrOp>(op)) {
          str = "or";
        } else if (isa<cggi::NorOp>(op)) {
          str = "nor";
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Try to parse boolean operation that does not exist.");
        }
        vectorizedGateOperands.push_back(StringAttr::get(&context, str));
      }

      // Group the independent operands over the operations
      for (uint operandIndex = 0; operandIndex < key->getNumOperands();
           ++operandIndex) {
        SmallVector<Value, 4> operands;
        LLVM_DEBUG({
          llvm::dbgs() << "For: " << key->getName()
                       << " Number of ops: " << key->getNumOperands() << "\n";
        });

        operands.reserve(bucket.size());
        ///------------------------------------------
        for (auto *op : bucket) {
          LLVM_DEBUG(llvm::dbgs() << "getOperand for [" << operandIndex << "]: "
                                  << op->getOperand(operandIndex) << "\n");
          operands.push_back(op->getOperand(operandIndex));
        }
        ///------------------------------------------

        auto fromElementsOp = builder.create<tensor::FromElementsOp>(
            key->getLoc(), tensorType, operands);
        vectorizedOperands.push_back(fromElementsOp.getResult());
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Go over vectorizedOps:\n";
        for (auto op : vectorizedOperands) {
          llvm::dbgs() << " - " << op << "\n";
        }
        llvm::dbgs() << "Go over vectorizedGateOps:\n";
        for (auto op : vectorizedGateOperands) {
          llvm::dbgs() << " - " << op;
        }
        llvm::dbgs() << "\n";
      });

      auto oplist = CGGIGateAttr::get(&context, vectorizedGateOperands);

      auto vectorizedOp = builder.create<cggi::PackedOp>(
          key->getLoc(), tensorType, oplist, vectorizedOperands[0],
          vectorizedOperands[1]);

      int bucketIndex = 0;

      for (auto *op : bucket) {
        auto extractionIndex = builder.create<arith::ConstantOp>(
            op->getLoc(), builder.getIndexAttr(bucketIndex));
        auto extractOp = builder.create<tensor::ExtractOp>(
            op->getLoc(), elementType, vectorizedOp->getResult(0),
            extractionIndex.getResult());
        op->replaceAllUsesWith(ValueRange{extractOp.getResult()});
        bucketIndex++;
      }

      for (auto *op : bucket) {
        op->erase();
      }
      madeReplacement = (bucketIndex > 0) || madeReplacement;
    }
  }

  return madeReplacement;
}

struct BooleanLineVectorizer
    : impl::BooleanLineVectorizerBase<BooleanLineVectorizer> {
  using BooleanLineVectorizerBase::BooleanLineVectorizerBase;

  void runOnOperation() override {
    MLIRContext &context = getContext();

    getOperation()->walk<WalkOrder::PreOrder>([&](Block *block) {
      if (tryBoolVectorizeBlock(block, context)) {
        sortTopologically(block);
      }
    });
  }
};

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
