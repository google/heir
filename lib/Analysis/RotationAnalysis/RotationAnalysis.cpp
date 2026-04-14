#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>

#include "lib/Analysis/RotationAnalysis/DagBuilder.h"
#include "lib/Analysis/RotationAnalysis/RotationEvalVisitor.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"     // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"       // from @llvm-project

#define DEBUG_TYPE "rotation-analysis"

namespace mlir {
namespace heir {

using Node = kernel::ArithmeticDagNode<kernel::LiteralValue>;
using NodePtr = std::shared_ptr<Node>;

LogicalResult RotationAnalysis::handleScfFor(scf::ForOp forOp) {
  // Ensure we have the outermost loop
  scf::ForOp outermostFor = forOp;
  while (auto parent = outermostFor->getParentOfType<scf::ForOp>()) {
    outermostFor = parent;
  }
  LDBG() << "Analyzing rotations in scf.for op: " << forOp;

  auto numSlotsAttr = dyn_cast_or_null<IntegerAttr>(
      forOp->getParentOfType<ModuleOp>()->getAttr(kActualSlotCountAttrName));
  if (!numSlotsAttr) {
    LDBG() << "Unable to determine the total number of slots when translating "
              "scf.for op: "
           << forOp;
    return failure();
  }

  DagBuilder dagBuilder(numSlotsAttr.getInt());
  FailureOr<NodePtr> res = dagBuilder.build(outermostFor.getOperation());

  if (failed(res)) return failure();

  NodePtr dag = res.value();
  auto shifts = evalRotations(dag);
  LDBG() << "Found new shifts: "
         << llvm::join(
                llvm::map_range(
                    shifts, [](int64_t dim) { return std::to_string(dim); }),
                ",");
  rotationIndices.insert(shifts.begin(), shifts.end());

  // All the rotation ops within the outermost for loop are analyzed.
  outermostFor->walk([&](RotationOpInterface rotOp) { markVisited(rotOp); });

  return success();
}

LogicalResult RotationAnalysis::analyzeRotationOp(
    RotationOpInterface rotationOp) {
  if (wasVisited(rotationOp)) {
    LDBG() << "Skipping already-visited rotation op "
           << *rotationOp.getOperation();
    return success();
  }
  LDBG() << "Analyzing rotation op " << *rotationOp.getOperation();

  // Handle cases where the rotation shift can be statically folded via constant
  // propagation.
  auto indices = rotationOp.getRotationIndices();
  SmallVector<int64_t> constantIndices;
  bool allConstant = true;
  for (auto& ofr : indices) {
    if (auto attr = dyn_cast_if_present<Attribute>(ofr)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        constantIndices.push_back(intAttr.getInt());
        continue;
      }
    }
    if (auto value = dyn_cast<Value>(ofr)) {
      IntegerAttr attr;
      if (matchPattern(value, m_Constant(&attr))) {
        constantIndices.push_back(attr.getInt());
        continue;
      }
    }
    allConstant = false;
    break;
  }
  if (allConstant) {
    LDBG() << "Rotation op has constant indices: "
           << *rotationOp.getOperation();
    rotationIndices.insert(constantIndices.begin(), constantIndices.end());
    markVisited(rotationOp);
    return success();
  }

  // Check if it's inside an scf loop. In this case, we reconstruct the loop as
  // an ArithmeticDag, and simulate its execution (ignoring all ops except those
  // involving scalars).
  Operation* rotOp = rotationOp.getOperation();
  if (auto scfFor = rotOp->getParentOfType<scf::ForOp>()) {
    LDBG() << "Rotation op is in an scf.for op";
    return handleScfFor(scfFor);
  }

  // TODO(#2712): Support affine.for op.
  if (auto affineForOp = rotOp->getParentOfType<affine::AffineForOp>()) {
    LDBG() << "affine.for op not supported in RotationAnalysis";
    return failure();
  }

  // If the rotation op is not in a loop, and not handled by folding,
  // then we need to see what types of IRs are needed to support here.
  llvm::errs() << "rotate op not supported in RotationAnalysis; "
               << "Please file a bug with the HEIR team with your IR. "
               << "Op was " << *rotationOp.getOperation() << "\nIR was: ";
  rotationOp.getOperation()->getParentOfType<func::FuncOp>()->dump();
  return failure();
}

LogicalResult RotationAnalysis::run(Operation* op) {
  WalkResult walkResult = op->walk([&](RotationOpInterface rotationOp) {
    if (failed(analyzeRotationOp(rotationOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return (walkResult.wasInterrupted()) ? failure() : success();
}

}  // namespace heir
}  // namespace mlir
