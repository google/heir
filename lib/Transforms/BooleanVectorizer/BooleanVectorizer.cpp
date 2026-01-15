#include "lib/Transforms/BooleanVectorizer/BooleanVectorizer.h"

#include <cassert>
#include <cstdint>
#include <vector>

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/TopologicalSortUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

#define DEBUG_TYPE "bool-vectorizer"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_BOOLEANVECTORIZER
#include "lib/Transforms/BooleanVectorizer/BooleanVectorizer.h.inc"

SmallVector<Value> buildVectorizedOperands(
    Operation* key, const SmallVector<Operation*>& bucket, OpBuilder builder) {
  SmallVector<Value> vectorizedOperands;
  // Group the independent operands over the operations - check if this is
  // operandIsMappable and if so, build a from elements op.
  for (uint operandIndex = 0; operandIndex < key->getNumOperands();
       ++operandIndex) {
    if (auto interface = dyn_cast<ElementwiseByOperandOpInterface>(key)) {
      // If the operand is not mappable, skip it.
      if (!interface.operandIsMappable(operandIndex)) {
        continue;
      }
    }

    SmallVector<Value, 4> operands;
    LLVM_DEBUG({
      llvm::dbgs() << "For: " << key->getName()
                   << " Number of ops: " << key->getNumOperands() << "\n";
    });

    operands.reserve(bucket.size());
    ///------------------------------------------
    for (auto* op : bucket) {
      LLVM_DEBUG(llvm::dbgs() << "getOperand for [" << operandIndex
                              << "]: " << op->getOperand(operandIndex) << "\n");
      operands.push_back(op->getOperand(operandIndex));
    }
    ///------------------------------------------
    RankedTensorType tensorType =
        RankedTensorType::get(bucket.size(), operands[0].getType());
    auto fromElementsOp = tensor::FromElementsOp::create(builder, key->getLoc(),
                                                         tensorType, operands);
    vectorizedOperands.push_back(fromElementsOp.getResult());
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Go over vectorizedOps:\n";
    for (auto op : vectorizedOperands) {
      llvm::dbgs() << " - " << op << "\n";
    }
  });
  return vectorizedOperands;
}

int bucketSize(const SmallVector<SmallVector<Operation*>>& buckets) {
  int size = 0;
  for (const auto& bucket : buckets) {
    size += bucket.size();
  }
  return size;
}

DenseMap<Operation*, SmallVector<SmallVector<Operation*>>> buildCompatibleOps(
    std::vector<mlir::Operation*> level, int parallelism) {
  DenseMap<Operation*, SmallVector<SmallVector<Operation*>>> compatibleOps;

  for (auto* op : level) {
    bool foundCompatible = false;

    for (auto& [key, buckets] : compatibleOps) {
      if (cast<BatchVectorizableOpInterface>(key).isBatchCompatible(op)) {
        if (parallelism == 0 ||
            compatibleOps[key].back().size() < parallelism) {
          compatibleOps[key].back().push_back(op);
        } else {
          SmallVector<Operation*> newBucket;
          newBucket.push_back(op);
          compatibleOps[key].push_back(newBucket);
        }
        foundCompatible = true;
      }
    }

    if (!foundCompatible) {
      SmallVector<Operation*> newBucket;
      newBucket.push_back(op);
      compatibleOps[op].push_back(newBucket);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Partitioned level of size " << level.size()
                          << " into " << compatibleOps.size()
                          << " groups of compatible ops\n");
  return compatibleOps;
}

bool tryBoolVectorizeBlock(Block* block, MLIRContext& context,
                           int parallelism) {
  graph::Graph<Operation*> graph;
  for (auto& op : block->getOperations()) {
    if (!isa<BatchVectorizableOpInterface>(op)) {
      continue;
    }

    graph.addVertex(&op);
    SetVector<Operation*> backwardSlice;
    BackwardSliceOptions options;
    options.omitBlockArguments = true;

    (void)mlir::getBackwardSlice(&op, &backwardSlice, options);
    for (auto* upstreamDep : backwardSlice) {
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
    for (const auto& level : levels) {
      llvm::dbgs() << "\nLevel " << level_num++ << ":\n";
      for (auto op : level) {
        llvm::dbgs() << " - " << *op << "\n";
      }
    }
  });

  bool madeReplacement = false;
  for (const auto& level : levels) {
    DenseMap<Operation*, SmallVector<SmallVector<Operation*>>> compatibleOps =
        buildCompatibleOps(level, parallelism);

    LLVM_DEBUG({
      llvm::dbgs()
          << " ########## Overview of the Compatible Ops object ##########\n";
      for (const auto& elem : compatibleOps) {
        llvm::dbgs() << " KEY " << *elem.first << "\n";
        for (const auto op : elem.getSecond()) {
          for (const auto opp : op) {
            llvm::dbgs() << " - " << *opp << "\n";
          }
          llvm::dbgs() << " next \n";
        }
      }
    });

    // Loop over all the compatibleOp groups
    // Each loop will have the key and a bucket with all the operations in
    for (const auto& [key, buckets] : compatibleOps) {
      if (bucketSize(buckets) < 2) {
        continue;
      }
      for (const auto& bucket : buckets) {
        LLVM_DEBUG({
          llvm::dbgs() << "[**START] Bucket (" << key->getName()
                       << ") \t Vectorizing ops:\n"
                       << *key << "\n";

          for (const auto op : bucket) {
            llvm::dbgs() << " - " << *op << "\n";
          }
        });

        OpBuilder builder(bucket.back());
        SmallVector<Value> vectorizedOperands =
            buildVectorizedOperands(key, bucket, builder);
        BatchVectorizableOpInterface keyInterface(key);
        FailureOr<Operation*> maybeVectorizedOp =
            keyInterface.buildBatchedOperation(builder.getContext(), builder,
                                               vectorizedOperands, bucket);
        if (failed(maybeVectorizedOp)) {
          return false;
        }
        Operation* vectorizedOp = maybeVectorizedOp.value();
        Type elementType =
            cast<RankedTensorType>(vectorizedOp->getResult(0).getType())
                .getElementType();

        int bucketIndex = 0;
        for (auto* op : bucket) {
          auto extractionIndex = arith::ConstantOp::create(
              builder, op->getLoc(), builder.getIndexAttr(bucketIndex));
          auto extractOp = tensor::ExtractOp::create(
              builder, op->getLoc(), elementType, vectorizedOp->getResult(0),
              extractionIndex.getResult());
          op->replaceAllUsesWith(ValueRange{extractOp.getResult()});
          bucketIndex++;
        }
        madeReplacement = (bucketIndex > 0) || madeReplacement;
      }
      // Erase Ops that have been replaced for a specific key.
      for (const auto& bucket : buckets) {
        for (auto* op : bucket) {
          op->erase();
        }
      }
    }
  }

  return madeReplacement;
}

struct BooleanVectorizer : impl::BooleanVectorizerBase<BooleanVectorizer> {
  using BooleanVectorizerBase::BooleanVectorizerBase;

  void runOnOperation() override {
    MLIRContext& context = getContext();

    getOperation()->walk<WalkOrder::PreOrder>([&](Block* block) {
      if (tryBoolVectorizeBlock(block, context, parallelism)) {
        sortTopologically(block);
      }
    });
  }
};

}  // namespace heir
}  // namespace mlir
