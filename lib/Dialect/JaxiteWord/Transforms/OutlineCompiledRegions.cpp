#include "lib/Dialect/JaxiteWord/Transforms/OutlineCompiledRegions.h"

#include <cstdint>

#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/ADT/SetVector.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallPtrSet.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                // from @llvm-project

namespace mlir {
namespace heir {
namespace jaxiteword {

#define GEN_PASS_DEF_OUTLINECOMPILEDREGIONS
#include "lib/Dialect/JaxiteWord/Transforms/Passes.h.inc"

namespace {

bool isEligible(Operation* op) {
  return isa<MulPlainOp, RotOp, AddOp, AddPlainOp, ModReduceOp, LevelReduceOp,
             MulNoRelinOp, RelinOp, MulOp, SquareOp>(op);
}

bool isCiphertext(Value value) {
  return isa<lwe::LWECiphertextType>(value.getType());
}

bool consumesAny(Operation* op, const llvm::SmallPtrSetImpl<Value>& values) {
  return llvm::any_of(op->getOperands(),
                      [&](Value value) { return values.contains(value); });
}

struct Candidate {
  SmallVector<Operation*> operations;
  llvm::SmallPtrSet<Value, 32> roots;
  llvm::SmallPtrSet<Value, 32> produced;
};

Candidate findCandidate(Block& block, int64_t maximumOperations) {
  Candidate candidate;
  for (Operation& operation : block.getOperations()) {
    Operation* op = &operation;
    if (isa<CompiledRegionOp>(op)) continue;

    if (!isEligible(op)) {
      if (!candidate.operations.empty() &&
          consumesAny(op, candidate.produced)) {
        break;
      }
      continue;
    }

    bool connected = candidate.operations.empty();
    for (Value operand : op->getOperands()) {
      if (!isCiphertext(operand)) continue;
      if (candidate.operations.empty()) {
        candidate.roots.insert(operand);
      } else if (candidate.roots.contains(operand) ||
                 candidate.produced.contains(operand)) {
        connected = true;
      }
    }
    if (!connected) break;
    if (static_cast<int64_t>(candidate.operations.size()) >=
        maximumOperations) {
      break;
    }

    candidate.operations.push_back(op);
    for (Value result : op->getResults()) {
      if (isCiphertext(result)) candidate.produced.insert(result);
    }
  }
  return candidate;
}

LogicalResult outlineCandidate(Candidate& candidate, int64_t regionId) {
  llvm::SmallPtrSet<Operation*, 32> selected(candidate.operations.begin(),
                                             candidate.operations.end());
  SetVector<Value> inputs;
  SetVector<Value> outputs;
  for (Operation* op : candidate.operations) {
    for (Value operand : op->getOperands()) {
      Operation* definingOp = operand.getDefiningOp();
      if (!definingOp || !selected.contains(definingOp)) inputs.insert(operand);
    }
    for (Value result : op->getResults()) {
      if (llvm::any_of(result.getUsers(), [&](Operation* user) {
            return !selected.contains(user);
          })) {
        outputs.insert(result);
      }
    }
  }
  if (outputs.empty()) return failure();

  Operation* last = candidate.operations.back();
  OpBuilder builder(last);
  builder.setInsertionPointAfter(last);
  SmallVector<Type> outputTypes =
      llvm::map_to_vector(outputs, [](Value value) { return value.getType(); });
  auto regionOp = CompiledRegionOp::create(builder, last->getLoc(), outputTypes,
                                           inputs.getArrayRef(), regionId);

  Region& body = regionOp.getBody();
  Block* block = new Block();
  body.push_back(block);
  IRMapping mapping;
  for (Value input : inputs) {
    mapping.map(input, block->addArgument(input.getType(), last->getLoc()));
  }
  builder.setInsertionPointToStart(block);
  for (Operation* op : candidate.operations) {
    builder.clone(*op, mapping);
  }
  SmallVector<Value> yielded;
  for (Value output : outputs) yielded.push_back(mapping.lookup(output));
  CompiledRegionYieldOp::create(builder, last->getLoc(), yielded);

  for (unsigned index = 0; index < outputs.size(); ++index) {
    Value output = outputs[index];
    output.replaceUsesWithIf(regionOp.getOutputs()[index], [&](OpOperand& use) {
      return !selected.contains(use.getOwner());
    });
  }
  for (Operation* op : llvm::reverse(candidate.operations)) op->erase();
  return success();
}

struct OutlineCompiledRegions
    : impl::OutlineCompiledRegionsBase<OutlineCompiledRegions> {
  using OutlineCompiledRegionsBase::OutlineCompiledRegionsBase;

  void runOnOperation() override {
    if (minimumOperations < 1 || maximumOperations < minimumOperations ||
        maximumRegions < 1) {
      getOperation()->emitError("invalid compiled-region size options");
      return signalPassFailure();
    }

    int64_t nextRegionId = 0;
    WalkResult result = getOperation()->walk([&](func::FuncOp function) {
      if (nextRegionId >= maximumRegions) return WalkResult::interrupt();
      if (!function.getBody().hasOneBlock()) return WalkResult::advance();
      while (nextRegionId < maximumRegions) {
        Candidate candidate =
            findCandidate(function.getBody().front(), maximumOperations);
        if (static_cast<int64_t>(candidate.operations.size()) <
            minimumOperations) {
          break;
        }
        if (failed(outlineCandidate(candidate, nextRegionId++))) {
          function.emitError("failed to outline compiled JaxiteWord region");
          return WalkResult::interrupt();
        }
      }
      return nextRegionId >= maximumRegions ? WalkResult::interrupt()
                                            : WalkResult::advance();
    });
    if (result.wasInterrupted() && nextRegionId < maximumRegions) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir
