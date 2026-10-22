#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"

#include <algorithm>
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
#include "lib/Utils/RotationUtils.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"     // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
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

namespace {
/// A linear combination + constant of SSA values
struct LinearExpr {
  DenseMap<Value, int64_t> coeffs;
  int64_t constant = 0;
};
}

static void accumulate(LinearExpr& dst, const LinearExpr& src, int64_t scale) {
  for (const auto& [value, coeff] : src.coeffs) {
    dst.coeffs[value] += scale * coeff;
  }
  dst.constant += scale * src.constant;
}

/// Decomposes `ofr` into a LinearExpr of non-constant leaf values 
static FailureOr<LinearExpr> getLinearExpr(OpFoldResult ofr) {
  LinearExpr expr;
  if (auto cst = getConstantIntValue(ofr)) {
    expr.constant = *cst;
    return expr;
  }

  auto value = dyn_cast<Value>(ofr);
  if (!value) return failure();

  if (auto addOp = value.getDefiningOp<arith::AddIOp>()) {
    FailureOr<LinearExpr> lhs = getLinearExpr(addOp.getLhs());
    FailureOr<LinearExpr> rhs = getLinearExpr(addOp.getRhs());
    if (failed(lhs) || failed(rhs)) return failure();
    accumulate(expr, *lhs, 1);
    accumulate(expr, *rhs, 1);
    return expr;
  }

  if (auto subOp = value.getDefiningOp<arith::SubIOp>()) {
    FailureOr<LinearExpr> lhs = getLinearExpr(subOp.getLhs());
    FailureOr<LinearExpr> rhs = getLinearExpr(subOp.getRhs());
    if (failed(lhs) || failed(rhs)) 
      return failure();
    accumulate(expr, *lhs, 1);
    accumulate(expr, *rhs, -1);
    return expr;
  }

  if (auto mulOp = value.getDefiningOp<arith::MulIOp>()) {
    auto lhsCst = getConstantIntValue(mulOp.getLhs());
    auto rhsCst = getConstantIntValue(mulOp.getRhs());
    if (!lhsCst && !rhsCst) return failure();
    int64_t scale = lhsCst ? *lhsCst : *rhsCst;
    FailureOr<LinearExpr> other =
        getLinearExpr(lhsCst ? mulOp.getRhs() : mulOp.getLhs());
    if (failed(other)) return failure();
    accumulate(expr, *other, scale);
    return expr;
  }

  expr.coeffs[value] = 1;
  return expr;
}

static LogicalResult addScfForOpDomain(affine::FlatAffineValueConstraints& cst,
                                scf::ForOp forOp) {
  auto lb = getConstantIntValue(forOp.getLowerBound());
  auto ub = getConstantIntValue(forOp.getUpperBound());
  auto step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step) return failure();

  Value iv = forOp.getInductionVar();
  unsigned ivPos;
  if (!cst.findVar(iv, &ivPos)) return failure();
  
  // Add lower and upper bound constraints
  cst.addBound(presburger::BoundType::LB, iv, *lb);
  cst.addBound(presburger::BoundType::UB, iv, *ub - 1);

  // Add constraints for the stride access
  if (*step > 1) {
    unsigned localPos = cst.appendVar(presburger::VarKind::Local);
    SmallVector<int64_t> eq(cst.getNumCols(), 0);
    eq[ivPos] = 1;
    eq[localPos] = -*step;
    eq.back() = -*lb;
    cst.addEquality(eq);
  }
  return success();
}

LogicalResult RotationAnalysis::analyzeRotationOpIfAffine(
    RotationOpInterface rotationOp) {
  Operation* op = rotationOp.getOperation();

  auto numSlotsAttr = dyn_cast_or_null<IntegerAttr>(
      op->getParentOfType<ModuleOp>()->getAttr(kActualSlotCountAttrName));
  if (!numSlotsAttr) {
    LDBG() << "Unable to determine the total number of slots "
           << op;
    return failure();
  }

  int64_t numSlots = numSlotsAttr.getInt();

  // Collect the enclosing scf.for ops
  SmallVector<scf::ForOp> loops;
  for (auto forOp = op->getParentOfType<scf::ForOp>(); forOp;
       forOp = forOp->getParentOfType<scf::ForOp>()) {
    loops.push_back(forOp);
  }
  if (loops.empty()) return failure();
  std::reverse(loops.begin(), loops.end());

  SmallVector<Value, 4> ivs;
  for (scf::ForOp forOp : loops) ivs.push_back(forOp.getInductionVar());

  affine::FlatAffineValueConstraints cst(ivs.size(), /*numSymbols=*/0,
                                         /*numLocals=*/0, ivs);
  for (scf::ForOp forOp : loops) {
    if (failed(addScfForOpDomain(cst, forOp))) {
      LDBG() << "Loop bounds and step must be constant " << forOp;
      return failure();
    }
  }

  // Add constraint for each linearized rotation index 
  unsigned numLoopDims = ivs.size();
  for (OpFoldResult ofr : rotationOp.getRotationIndices()) {
    FailureOr<LinearExpr> linearizedExpr = getLinearExpr(ofr);
    if (failed(linearizedExpr)) {
      LDBG() << "Rotation index is not a linear expression " << ofr;
      return failure();
    }

    // Add `shift - linearizedExpr = 0` constraint
    unsigned shiftPos = cst.appendVar(presburger::VarKind::SetDim);
    SmallVector<int64_t> eq(cst.getNumCols(), 0);
    eq[shiftPos] = 1;
    for (const auto& [leafValue, coeff] : linearizedExpr->coeffs) {
      unsigned pos;
      if (!cst.findVar(leafValue, &pos)) {
        LDBG() << "Rotation index depends on a value that is not an scf.for induction variable "
               << leafValue;
        return failure();
      }
      eq[pos] -= coeff;
    }
    eq.back() = -linearizedExpr->constant;
    cst.addEquality(eq);
  }

  // Convert loop IVs to Domain kind, so getRangePoints can project them out
  presburger::IntegerRelation rel(cst);
  if (numLoopDims > 0) {
    rel.convertVarKind(presburger::VarKind::Range, 0, numLoopDims,
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

  // Handle rotation indices that are linear functions of the induction
  // variables of scf loops with constant bounds and steps
  if (succeeded(analyzeRotationOpIfAffine(rotationOp))) {
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
