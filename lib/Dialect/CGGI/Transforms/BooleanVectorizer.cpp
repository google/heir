#include "lib/Dialect/CGGI/Transforms/BooleanVectorizer.h"

#include <cassert>
#include <cstdint>
#include <vector>

#include "lib/Dialect/CGGI/IR/CGGIAttributes.h"
#include "lib/Dialect/CGGI/IR/CGGIEnums.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
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
namespace cggi {

#define GEN_PASS_DEF_BOOLEANVECTORIZER
#include "lib/Dialect/CGGI/Transforms/Passes.h.inc"

bool areCompatibleBool(Operation* lhs, Operation* rhs) {
  if (lhs->getDialect() != rhs->getDialect() ||
      lhs->getResultTypes() != rhs->getResultTypes() ||
      lhs->getNumOperands() != rhs->getNumOperands() ||
      // Attributes do not need to match for Lut3
      (!llvm::isa<cggi::Lut3Op>(lhs) && lhs->getAttrs() != rhs->getAttrs()) ||
      isa<NotOp>(rhs) || isa<NotOp>(lhs)) {
    return false;
  }
  // TODO: Check if can be made better with a BooleanPackableGate trait
  // on the CGGI_BinaryGateOp's?
  return OpTrait::hasElementwiseMappableTraits(lhs);
}

FailureOr<SmallVector<Attribute>> buildGateOperands(
    const SmallVector<Operation*>& bucket, MLIRContext& context) {
  SmallVector<Attribute> vectorizedGateOperands;
  for (auto* op : bucket) {
    FailureOr<Attribute> attr =
        llvm::TypeSwitch<Operation&, FailureOr<Attribute>>(*op)
            .Case<cggi::AndOp>([&context](AndOp op) {
              return CGGIBoolGateEnumAttr::get(&context, CGGIBoolGateEnum::AND);
            })
            .Case<cggi::NandOp>([&context](NandOp op) {
              return CGGIBoolGateEnumAttr::get(&context,
                                               CGGIBoolGateEnum::NAND);
            })
            .Case<cggi::XorOp>([&context](XorOp op) {
              return CGGIBoolGateEnumAttr::get(&context, CGGIBoolGateEnum::XOR);
            })
            .Case<cggi::XNorOp>([&context](XNorOp op) {
              return CGGIBoolGateEnumAttr::get(&context,
                                               CGGIBoolGateEnum::XNOR);
            })
            .Case<cggi::OrOp>([&context](OrOp op) {
              return CGGIBoolGateEnumAttr::get(&context, CGGIBoolGateEnum::OR);
            })
            .Case<cggi::NorOp>([&context](NorOp op) {
              return CGGIBoolGateEnumAttr::get(&context, CGGIBoolGateEnum::NOR);
            })
            .Case<cggi::NotOp>([&context](NotOp op) {
              return CGGIBoolGateEnumAttr::get(&context, CGGIBoolGateEnum::NOT);
            })
            .Case<cggi::Lut3Op>(
                [&](cggi::Lut3Op op) { return op.getLookupTable(); })
            .Default([&](Operation& op) -> FailureOr<Attribute> {
              return failure();
            });
    if (failed(attr)) {
      op->emitOpError("unsupported operation for vectorization");
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

SmallVector<Value> buildVectorizedOperands(
    Operation* key, const SmallVector<Operation*>& bucket,
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
    for (auto* op : bucket) {
      LLVM_DEBUG(llvm::dbgs() << "getOperand for [" << operandIndex
                              << "]: " << op->getOperand(operandIndex) << "\n");
      operands.push_back(op->getOperand(operandIndex));
    }
    ///------------------------------------------
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
      if (areCompatibleBool(key, op) || (isa<NotOp>(key) && isa<NotOp>(op))) {
        if (parallelism == 0 ||
            compatibleOps[key].back().size() < parallelism) {
          compatibleOps[key].back().push_back(op);
        } else {
          SmallVector<Operation*> newBucket;
          newBucket.push_back(op);
          compatibleOps[key].push_back(newBucket);
        }
        foundCompatible = true;

      } else if (isa<NotOp>(op) &&
                 llvm::count_if(compatibleOps, [](const auto& pair) {
                   return isa<cggi::NotOp>(pair.first);
                 }) == 0) {
        SmallVector<Operation*> newBucket;
        newBucket.push_back(op);
        compatibleOps[op].push_back(newBucket);
        foundCompatible = true;
        break;  // Now we extend the compatible ops when looping over the
                // structure: Possible to now iterate over the newly created key
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
    if (!op.hasTrait<OpTrait::Elementwise>()) {
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
        // relies on CGGI ops having a single result type
        Type elementType = key->getResultTypes()[0];
        RankedTensorType tensorType = RankedTensorType::get(
            {static_cast<int64_t>(bucket.size())}, elementType);

        SmallVector<Value> vectorizedOperands =
            buildVectorizedOperands(key, bucket, tensorType, builder);
        auto vectorizedGateOperands = buildGateOperands(bucket, context);
        if (failed(vectorizedGateOperands)) return false;

        Operation* vectorizedOp;
        if (llvm::isa<cggi::Lut3Op>(key)) {
          auto oplist = builder.getArrayAttr(vectorizedGateOperands.value());
          vectorizedOp = cggi::PackedLut3Op::create(
              builder, key->getLoc(), tensorType, oplist, vectorizedOperands[0],
              vectorizedOperands[1], vectorizedOperands[2]);
        } else if (llvm::isa<cggi::NotOp>(key)) {
          vectorizedOp = cggi::NotOp::create(builder, key->getLoc(), tensorType,
                                             vectorizedOperands[0]);
        } else {
          auto operands = vectorizedGateOperands.value();
          auto oplist = CGGIBoolGatesAttr::get(
              &context,
              llvm::to_vector(llvm::map_range(
                  operands, [](Attribute attr) -> CGGIBoolGateEnumAttr {
                    return cast<CGGIBoolGateEnumAttr>(attr);
                  })));
          vectorizedOp = cggi::PackedOp::create(
              builder, key->getLoc(), tensorType, oplist, vectorizedOperands[0],
              vectorizedOperands[1]);
        }

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

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
