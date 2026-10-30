#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "lib/Analysis/RotationAnalysis/DagBuilder.h"
#include "lib/Analysis/RotationAnalysis/RotationEvalVisitor.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/AffineLoopUtils.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/RotationUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"       // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"       // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"     // from @llvm-project

#define DEBUG_TYPE "rotation-analysis"

namespace mlir {
namespace heir {

using Node = kernel::ArithmeticDagNode<kernel::LiteralValue>;
using NodePtr = std::shared_ptr<Node>;

LogicalResult RotationAnalysis::analyzeRotationOpIfAffine(
    RotationOpInterface rotationOp) {
  Operation* op = rotationOp.getOperation();

  auto numSlotsAttr = dyn_cast_or_null<IntegerAttr>(
      op->getParentOfType<ModuleOp>()->getAttr(kActualSlotCountAttrName));
  if (!numSlotsAttr) {
    LDBG() << "Unable to determine the total number of slots " << op;
    return failure();
  }

  int64_t numSlots = numSlotsAttr.getInt();

  // Collect enclosing loops and affine.if ops, bail out on scf.if
  SmallVector<LoopLikeOpInterface> loops;
  SmallVector<EnclosingAffineIf> affineIfs;
  if (failed(getEnclosingLoopsAndIfs(op, loops, affineIfs)) || loops.empty())
    return failure();

  FailureOr<affine::FlatAffineValueConstraints> domain =
      getIterationDomain(loops, affineIfs);
  if (failed(domain)) return failure();
  affine::FlatAffineValueConstraints cst = std::move(*domain);
  SmallVector<Value> ivs =
      llvm::map_to_vector(loops, [](LoopLikeOpInterface loop) -> Value {
        return *loop.getSingleInductionVar();
      });

  // Add constraint for each rotation index
  unsigned numDomainDims = cst.getNumDimVars();
  for (OpFoldResult ofr : rotationOp.getRotationIndices()) {
    unsigned shiftPos = cst.appendDimVar();
    if (failed(addAffineBound(cst, presburger::BoundType::EQ, shiftPos, ofr,
                              ivs))) {
      return failure();
    }
  }

  // Convert loop IVs to Domain kind, so getRangePoints can project them out
  presburger::IntegerRelation rel(cst);
  if (numDomainDims > 0) {
    rel.convertVarKind(presburger::VarKind::Range, 0, numDomainDims,
                       presburger::VarKind::Domain);
  }

  PointCollector collector;
  getRangePoints(rel, collector);
  for (const auto& point : collector.points) {
    for (int64_t shift : point) {
      rotationIndices.insert(normalizeRotation(shift, numSlots));
    }
  }
  return success();
}

LogicalResult RotationAnalysis::handleLoop(LoopLikeOpInterface loop) {
  // Ensure we have the outermost loop
  LoopLikeOpInterface outermostLoop = loop;
  while (auto parent = outermostLoop->getParentOfType<LoopLikeOpInterface>()) {
    outermostLoop = parent;
  }
  LDBG() << "Analyzing rotations in loop op: " << loop;

  auto numSlotsAttr = dyn_cast_or_null<IntegerAttr>(
      loop->getParentOfType<ModuleOp>()->getAttr(kActualSlotCountAttrName));
  if (!numSlotsAttr) {
    LDBG() << "Unable to determine the total number of slots when translating "
              "loop op: "
           << loop;
    return failure();
  }

  DagBuilder dagBuilder(numSlotsAttr.getInt());
  FailureOr<NodePtr> res = dagBuilder.build(outermostLoop.getOperation());

  if (failed(res)) return failure();

  NodePtr dag = res.value();
  auto shifts = evalRotations(dag);
  LDBG() << "Found new shifts: "
         << llvm::join(
                llvm::map_range(
                    shifts, [](int64_t dim) { return std::to_string(dim); }),
                ",");
  rotationIndices.insert(shifts.begin(), shifts.end());

  // All the rotation ops within the outermost loop are analyzed.
  outermostLoop->walk([&](RotationOpInterface rotOp) { markVisited(rotOp); });

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

  // Attempt to statically infer rotation indices that are affine
  // functions of enclosing loop induction variables
  if (succeeded(analyzeRotationOpIfAffine(rotationOp))) {
    markVisited(rotationOp);
    return success();
  }

  // If the static analysis failed, and the op is inside a loop, fall back to
  // dynamic analysis. In this case, we reconstruct the loop as an
  // ArithmeticDag, and simulate its execution.
  Operation* rotOp = rotationOp.getOperation();
  if (auto loop = rotOp->getParentOfType<LoopLikeOpInterface>()) {
    LDBG() << "Rotation op is in a loop op";
    return handleLoop(loop);
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
