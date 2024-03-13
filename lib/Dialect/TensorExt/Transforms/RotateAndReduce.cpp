#include "include/Dialect/TensorExt/Transforms/RotateAndReduce.h"

#include <cstdint>

#include "include/Analysis/RotationAnalysis/RotationAnalysis.h"
#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/TensorExt/IR/TensorExtOps.h"
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

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_ROTATEANDREDUCE
#include "include/Dialect/TensorExt/Transforms/Passes.h.inc"

/// A pass that searches for a length N sequence of binary operations that
/// reduces a length N vector to a single scalar, and replaces it with a
/// logarithmic number of rotations and binary operations.
struct RotateAndReduce : impl::RotateAndReduceBase<RotateAndReduce> {
  using RotateAndReduceBase::RotateAndReduceBase;

  template <typename ArithOp>
  void tryReplaceRotations(ArithOp op, Value tensor,
                           DenseSet<Operation *> &visited,
                           DataFlowSolver &solver) {
    // The dataflow analysis provides some guarantees, but not enough
    // to prove that we can replace the op with the rotate-and-reduce trick
    // while still maintaining program correctness.
    //
    // We need to do some more complicated checks to ensure that: the op tree
    // all contains the same op type (all sum or all mul), and that the
    // accessed rotations are included only once in the reduction.
    // This cannot be done during the dataflow analysis itself due to the
    // monotonicity requirements of the framework.
    LLVM_DEBUG(llvm::dbgs()
               << "Trying to replace rotations ending in " << *op << "\n");
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions options;
    // asserts that the parent op has a single region with a single block.
    options.omitBlockArguments = false;

    DenseSet<Operation *> visitedReductionOps;
    DenseMap<llvm::StringRef, int> opCounts;
    opCounts[op->getName().getStringRef()]++;

    getBackwardSlice(op.getOperation(), &backwardSlice, options);

    for (Operation *upstreamOpPtr : backwardSlice) {
      auto result =
          llvm::TypeSwitch<Operation *, LogicalResult>(upstreamOpPtr)
              .Case<arith::ConstantOp, tensor_ext::RotateOp>(
                  [&](auto upstreamOp) { return success(); })
              // Ignore generic ops
              .template Case<secret::GenericOp>(
                  [&](auto upstreamOp) { return success(); })
              .template Case<arith::AddIOp, arith::MulIOp>([&](auto
                                                                   upstreamOp) {
                opCounts[upstreamOp->getName().getStringRef()]++;
                // More than one reduction op is mixed in the reduction.
                if (opCounts.size() > 1) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "Not replacing op because reduction "
                                "contains multiple incompatible ops "
                             << op->getName() << " and "
                             << upstreamOp->getName() << "\n");
                  return failure();
                }

                // Inspect the lattice values at the join point,
                // and fail if there is any overlap
                auto *lhsLattice =
                    solver.lookupState<rotation_analysis::RotationLattice>(
                        upstreamOp.getLhs());
                auto *rhsLattice =
                    solver.lookupState<rotation_analysis::RotationLattice>(
                        upstreamOp.getRhs());
                LLVM_DEBUG(llvm::dbgs()
                           << "Computing overlap of "
                           << "lhs: " << lhsLattice->getValue() << "\n"
                           << "rhs: " << rhsLattice->getValue() << "\n");
                auto mergedLattice = rotation_analysis::RotationSets::overlap(
                    lhsLattice->getValue(), rhsLattice->getValue());
                LLVM_DEBUG(llvm::dbgs()
                           << "Overlap is: " << mergedLattice << "\n");
                if (!mergedLattice.empty()) {
                  LLVM_DEBUG(
                      llvm::dbgs()
                      << "Not replacing op because reduction "
                         "may not be a simple reduction of the input tensor\n"
                      << "lhs: " << lhsLattice->getValue() << "\n"
                      << "rhs: " << rhsLattice->getValue() << "\n");
                  return failure();
                }

                visitedReductionOps.insert(upstreamOp);
                return success();
              })
              .Default([&](Operation *op) {
                LLVM_DEBUG(llvm::dbgs() << "Not continuing because type switch "
                                           "encountered unsupported op "
                                        << op->getName() << "\n");
                return failure();
              });

      if (failed(result)) {
        return;
      }
    }

    // From here we know we will succeed.
    auto b = ImplicitLocOpBuilder(op->getLoc(), op);
    Operation *finalOp;
    auto tensorShape = tensor.getType().cast<RankedTensorType>().getShape();
    for (int64_t shiftSize = tensorShape[0] / 2; shiftSize > 0;
         shiftSize /= 2) {
      auto rotatedTensor = b.create<tensor_ext::RotateOp>(
          tensor, b.create<arith::ConstantOp>(b.getIndexAttr(shiftSize)));
      auto addOp = b.create<ArithOp>(tensor, rotatedTensor);
      finalOp = addOp;
      tensor = addOp->getResult(0);
    }

    [[maybe_unused]] auto *parentOp = op->getParentOp();
    op->replaceAllUsesWith(finalOp);
    LLVM_DEBUG(llvm::dbgs() << "Post-replacement: " << *parentOp << "\n");

    // Mark all ops in the reduction as visited so we don't try to replace them
    // twice.
    for (Operation *visitedOp : visitedReductionOps) {
      visited.insert(visitedOp);
    }
  }

  template <typename ArithOp>
  void tryReplaceExtractions(ArithOp op, DenseSet<Operation *> &visited) {
    LLVM_DEBUG(llvm::dbgs()
               << "Trying to replace extractions ending in " << *op << "\n");
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions options;
    // asserts that the parent op has a single region with a single block.
    options.omitBlockArguments = false;

    DenseSet<Value> inputTensors;
    DenseSet<Operation *> visitedReductionOps;
    DenseSet<unsigned> accessIndices;
    DenseMap<llvm::StringRef, int> opCounts;
    opCounts[op->getName().getStringRef()]++;

    // TODO(#523): replace backward slice with a dataflow analysis
    getBackwardSlice(op.getOperation(), &backwardSlice, options);
    for (Operation *upstreamOpPtr : backwardSlice) {
      auto result =
          llvm::TypeSwitch<Operation *, LogicalResult>(upstreamOpPtr)
              .Case<arith::ConstantOp>(
                  [&](auto upstreamOp) { return success(); })
              // Ignore generic ops
              .template Case<secret::GenericOp>(
                  [&](auto upstreamOp) { return success(); })
              .template Case<arith::AddIOp, arith::MulIOp>(
                  [&](auto upstreamOp) {
                    opCounts[upstreamOp->getName().getStringRef()]++;
                    // More than one reduction op is mixed in the reduction.
                    if (opCounts.size() > 1) {
                      LLVM_DEBUG(llvm::dbgs()
                                 << "Not replacing op because reduction "
                                    "contains multiple incompatible ops "
                                 << op->getName() << " and "
                                 << upstreamOp->getName() << "\n");
                      return failure();
                    }

                    // TODO(#522): support these non-tensor-extract operands by
                    // saving the values, and applying them again to the final
                    // result.
                    for (Value operand : upstreamOp->getOperands()) {
                      if (operand.getDefiningOp<arith::ConstantOp>()) {
                        LLVM_DEBUG(llvm::dbgs()
                                   << "Not replacing op because reduction "
                                      "includes non-tensor value operands "
                                   << operand << "\n");
                        return failure();
                      }
                    }
                    visitedReductionOps.insert(upstreamOp);
                    return success();
                  })
              .template Case<tensor::ExtractOp>([&](auto tensorOp) {
                inputTensors.insert(tensorOp.getTensor());
                if (inputTensors.size() > 1) {
                  LLVM_DEBUG(
                      llvm::dbgs()
                      << "Not replacing op due to multiple input tensors\n");
                  return failure();
                }

                // If the tensor is not 1D, we can't replace it with a rotate.
                if (tensorOp.getIndices().size() != 1) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "Not replacing op due to >1D input tensor\n");
                  return failure();
                }

                // If the access index is not constant, we can't tell if we are
                // reducing the entire vector (each index occurs exactly once in
                // the redution).
                arith::ConstantOp indexConstant =
                    tensorOp.getIndices()
                        .front()
                        .template getDefiningOp<arith::ConstantOp>();
                if (!indexConstant) {
                  LLVM_DEBUG(
                      llvm::dbgs()
                      << "Not replacing op due to non constant index access;"
                      << " (do you need to run --canonicalize or --sccp?)\n");
                  return failure();
                }
                int64_t accessIndex =
                    indexConstant.getValue().cast<IntegerAttr>().getInt();

                // If the access index was already seen, then fail because some
                // tensor element contributes more than once to the reduction.
                if (accessIndices.count(accessIndex)) {
                  LLVM_DEBUG(
                      llvm::dbgs()
                      << "Not replacing op because input tensor was accessed "
                         "multiple times in at same index\n");
                  return failure();
                }
                LLVM_DEBUG(llvm::dbgs()
                           << "Adding valid index " << accessIndex << "\n");
                accessIndices.insert(accessIndex);
                return success();
              })
              .Default([&](Operation *op) {
                LLVM_DEBUG(llvm::dbgs() << "Not continuing because type switch "
                                           "encountered unsupported op "
                                        << op->getName() << "\n");
                return failure();
              });

      if (failed(result)) {
        return;
      }
    }

    // The test for a match is now: does the number of accessed indices exactly
    // match the size of the tensor? I.e., does each tensor element show up
    // exactly once in the reduction?
    auto tensorShape =
        inputTensors.begin()->getType().cast<RankedTensorType>().getShape();
    if (tensorShape.size() != 1 || tensorShape[0] != accessIndices.size()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Not replacing op because tensor shape ("
                 << inputTensors.begin()->getType()
                 << ") is not fully reduced. Only " << accessIndices.size()
                 << " of " << tensorShape[0] << " indices were accessed\n");
      return;
    }

    // From here we know we will succeed.
    auto b = ImplicitLocOpBuilder(op->getLoc(), op);
    Value inputTensor = *inputTensors.begin();
    Operation *finalOp;
    for (int64_t shiftSize = tensorShape[0] / 2; shiftSize > 0;
         shiftSize /= 2) {
      auto rotatedTensor = b.create<tensor_ext::RotateOp>(
          inputTensor, b.create<arith::ConstantOp>(b.getIndexAttr(shiftSize)));
      auto addOp = b.create<ArithOp>(inputTensor, rotatedTensor);
      finalOp = addOp;
      inputTensor = addOp->getResult(0);
    }

    [[maybe_unused]] auto *parentOp = op->getParentOp();
    // We can extract at any index; every index contains the same reduced value.
    auto extractOp = b.create<tensor::ExtractOp>(
        finalOp->getResult(0), b.create<arith::ConstantIndexOp>(0).getResult());
    op->replaceAllUsesWith(extractOp);
    LLVM_DEBUG(llvm::dbgs() << "Post-replacement: " << *parentOp << "\n");

    // Mark all ops in the reduction as visited so we don't try to replace them
    // twice.
    for (Operation *visitedOp : visitedReductionOps) {
      visited.insert(visitedOp);
    }
  }

  void runOnOperation() override {
    DataFlowSolver solver;
    // These two upstream analyses are required dependencies for any sparse
    // dataflow analysis, or else the analysis will be a no-op. Cf.
    // https://github.com/llvm/llvm-project/issues/58922
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<rotation_analysis::RotationAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run dataflow analysis.\n";
      signalPassFailure();
      return;
    }

    LLVM_DEBUG({
      getOperation()->walk([&](Operation *op) {
        if (op->getNumResults() == 0) return;
        auto *targetSlotLattice =
            solver.lookupState<rotation_analysis::RotationLattice>(
                op->getResult(0));
        if (targetSlotLattice->getValue().isOverdetermined()) {
          llvm::dbgs() << "Rotation lattice for " << *op
                       << " is overdetermined\n";
        } else if (targetSlotLattice->getValue().empty()) {
          llvm::dbgs() << "Rotation lattice for " << *op << " is empty\n";
        } else {
          SmallVector<int64_t> sortedRotations(
              targetSlotLattice->getValue().getAccessedIndices().begin(),
              targetSlotLattice->getValue().getAccessedIndices().end());
          llvm::sort(sortedRotations);
          std::string stringified = llvm::join(
              llvm::map_range(sortedRotations,
                              [](int64_t i) { return std::to_string(i); }),
              ",");
          llvm::dbgs() << "Rotation lattice for " << *op << ": " << stringified
                       << "\n";
        }
      });
    });

    DenseSet<Operation *> visited;

    getOperation()->walk<WalkOrder::PreOrder, ReverseIterator>(
        [&](Operation *op) {
          if (op->getNumResults() == 0) return;
          auto *targetSlotLattice =
              solver.lookupState<rotation_analysis::RotationLattice>(
                  op->getResult(0));
          if (targetSlotLattice->getValue().isUninitialized() ||
              targetSlotLattice->getValue().isOverdetermined()) {
            return;
          }

          auto tensor = targetSlotLattice->getValue().getTensor();
          auto accessIndices =
              targetSlotLattice->getValue().getAccessedIndices();
          int64_t tensorSize =
              tensor.getType().cast<RankedTensorType>().getShape()[0];
          if (accessIndices.size() == tensorSize) {
            llvm::TypeSwitch<Operation &>(*op)
                .Case<arith::AddIOp>([&](auto arithOp) {
                  tryReplaceRotations<arith::AddIOp>(arithOp, tensor, visited,
                                                     solver);
                })
                .Case<arith::MulIOp>([&](auto arithOp) {
                  tryReplaceRotations<arith::MulIOp>(arithOp, tensor, visited,
                                                     solver);
                });
          }
        });

    // Traverse the IR in reverse order so that we can eagerly compute backward
    // slices for each operation.
    getOperation()->walk<WalkOrder::PreOrder, ReverseIterator>(
        [&](Operation *op) {
          if (visited.count(op)) {
            return;
          }
          llvm::TypeSwitch<Operation &>(*op)
              .Case<arith::AddIOp>([&](auto arithOp) {
                tryReplaceExtractions<arith::AddIOp>(arithOp, visited);
              })
              .Case<arith::MulIOp>([&](auto arithOp) {
                tryReplaceExtractions<arith::MulIOp>(arithOp, visited);
              });
        });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
