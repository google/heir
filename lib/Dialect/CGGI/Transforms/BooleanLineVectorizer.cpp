#include "lib/Dialect/CGGI/Transforms/BooleanLineVectorizer.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "lib/Dialect/CGGI/IR/CGGIAttributes.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Graph/Graph.h"
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
      lhs->getNumOperands() != rhs->getNumOperands() ||
      // Attributes do not need to match for Lut3
      (!llvm::isa<cggi::Lut3Op>(lhs) && lhs->getAttrs() != rhs->getAttrs())) {
    return false;
  }
  // TODO: Check if can be made better with a BooleanPackableGate trait
  // on the CGGI_BinaryGateOp's?
  return OpTrait::hasElementwiseMappableTraits(lhs);
}

FailureOr<SmallVector<Attribute>> BuildGateOperands(
    const SmallVector<Operation *> &bucket, MLIRContext &context) {
  SmallVector<Attribute> vectorizedGateOperands;
  for (auto *op : bucket) {
    FailureOr<Attribute> attr =
        llvm::TypeSwitch<Operation &, FailureOr<Attribute>>(*op)
            .Case<cggi::AndOp>(
                [&](auto op) { return StringAttr::get(&context, "and"); })
            .Case<cggi::NandOp>(
                [&](auto op) { return StringAttr::get(&context, "nand"); })
            .Case<cggi::XorOp>(
                [&](auto op) { return StringAttr::get(&context, "xor"); })
            .Case<cggi::XNorOp>(
                [&](auto op) { return StringAttr::get(&context, "xnor"); })
            .Case<cggi::Lut3Op>(
                [&](cggi::Lut3Op op) { return op.getLookupTable(); })
            .Default([&](Operation &op) -> FailureOr<Attribute> {
              op.emitOpError("Unable to find printer for op");
              return failure();
            });
    if (failed(attr)) {
      op->emitOpError("Failed attr");
      return failure();
    }
    vectorizedGateOperands.push_back(attr.value());
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Go over vectorizedGateOps:\n";
    for (auto op : vectorizedGateOperands) {
      llvm::dbgs() << " - " << op;
    }
    llvm::dbgs() << "\n";
  });
  return vectorizedGateOperands;
}

SmallVector<Value> BuildVectorizedOperands(
    Operation *key, const SmallVector<Operation *> &bucket,
    RankedTensorType tensorType, OpBuilder builder) {
  SmallVector<Value> vectorizedOperands;
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
      LLVM_DEBUG(llvm::dbgs() << "getOperand for [" << operandIndex
                              << "]: " << op->getOperand(operandIndex) << "\n");
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
  });
  return vectorizedOperands;
}

int bucket_size(const SmallVector<SmallVector<Operation *>> &buckets) {
  int size = 0;
  for (const auto &bucket : buckets) {
    size += bucket.size();
  }
  return size;
}

DenseMap<Operation *, SmallVector<SmallVector<Operation *>>> buildCompatibleOps(
    std::vector<mlir::Operation *> level, int parallelism) {
  DenseMap<Operation *, SmallVector<SmallVector<Operation *>>> compatibleOps;
  for (auto *op : level) {
    bool foundCompatible = false;
    for (auto &[key, buckets] : compatibleOps) {
      if (areCompatibleBool(key, op)) {
        if (parallelism == 0 ||
            compatibleOps[key].back().size() < parallelism) {
          compatibleOps[key].back().push_back(op);
        } else {
          SmallVector<Operation *> new_bucket;
          new_bucket.push_back(op);
          compatibleOps[key].push_back(new_bucket);
        }
        foundCompatible = true;
      }
    }
    if (!foundCompatible) {
      SmallVector<Operation *> new_bucket;
      new_bucket.push_back(op);
      compatibleOps[op].push_back(new_bucket);
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Partitioned level of size " << level.size()
                          << " into " << compatibleOps.size()
                          << " groups of compatible ops\n");
  return compatibleOps;
}

bool tryBoolVectorizeBlock(Block *block, MLIRContext &context,
                           int parallelism) {
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
    DenseMap<Operation *, SmallVector<SmallVector<Operation *>>> compatibleOps =
        buildCompatibleOps(level, parallelism);

    // Loop over all the compatibleOp groups
    // Each loop will have the key and a bucket with all the operations in
    for (const auto &[key, buckets] : compatibleOps) {
      if (bucket_size(buckets) < 2) {
        continue;
      }
      for (const auto &bucket : buckets) {
        LLVM_DEBUG({
          llvm::dbgs() << "[**START] Bucket \t Vectorizing ops:\n";
          for (const auto op : bucket) {
            llvm::dbgs() << " - " << *op << "\n";
          }
        });

        OpBuilder builder(bucket.back());
        // relies on CGGI ops having a single result type
        Type elementType = key->getResultTypes()[0];
        RankedTensorType tensorType = RankedTensorType::get(
            {static_cast<int64_t>(bucket.size())}, elementType);

        SmallVector<Value> vectorizedOperands =
            BuildVectorizedOperands(key, bucket, tensorType, builder);
        auto vectorizedGateOperands = BuildGateOperands(bucket, context);
        if (failed(vectorizedGateOperands)) return false;

        Operation *vectorizedOp;
        if (llvm::isa<cggi::Lut3Op>(key)) {
          auto oplist = builder.getArrayAttr(vectorizedGateOperands.value());
          vectorizedOp = builder.create<cggi::PackedLut3Op>(
              key->getLoc(), tensorType, oplist, vectorizedOperands[0],
              vectorizedOperands[1], vectorizedOperands[2]);
        } else {
          auto operands = ArrayRef<Attribute>(vectorizedGateOperands.value());
          auto oplist = CGGIGateAttr::get(
              &context, llvm::to_vector(llvm::map_range(
                            operands, [](Attribute attr) -> StringAttr {
                              return cast<StringAttr>(attr);
                            })));
          vectorizedOp = builder.create<cggi::PackedOp>(
              key->getLoc(), tensorType, oplist, vectorizedOperands[0],
              vectorizedOperands[1]);
        }

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
        madeReplacement = (bucketIndex > 0) || madeReplacement;
      }
      // Erase Ops that have been replaced for a specific key.
      for (const auto &bucket : buckets) {
        for (auto *op : bucket) {
          op->erase();
        }
      }
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
      if (tryBoolVectorizeBlock(block, context, parallelism)) {
        sortTopologically(block);
      }
    });
  }
};

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
