#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "lib/Analysis/RotationAnalysis/DagBuilder.h"
#include "lib/Analysis/RotationAnalysis/RotationEvalVisitor.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
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
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"         // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"           // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/ValueBoundsOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"     // from @llvm-project

#define DEBUG_TYPE "rotation-analysis"

namespace mlir {
namespace heir {

using Node = kernel::ArithmeticDagNode<kernel::LiteralValue>;
using NodePtr = std::shared_ptr<Node>;

static LogicalResult addAffineBound(affine::FlatAffineValueConstraints& cst,
                                    presburger::BoundType type, unsigned pos,
                                    OpFoldResult ofr, ArrayRef<Value> ivs,
                                    const ValueDimList& ivDeps) {
  if (std::optional<int64_t> c = getConstantIntValue(ofr)) {
    int64_t bound = (type == presburger::BoundType::UB) ? *c - 1 : *c;
    cst.addBound(type, pos, bound);
    return success();
  }

  Value val = cast<Value>(ofr);
  AffineMap map;
  SmallVector<Value> operands;
  if (llvm::is_contained(ivs, val)) {
    map = AffineMap::getMultiDimIdentityMap(1, val.getContext());
    operands.push_back(val);
  } else {
    // Walk the use-def chain via ValueBoundsOpInterface and return an
    // AffineMap of the IVs.
    ValueDimList mapOperands;
    if (failed(ValueBoundsConstraintSet::computeDependentBound(
            map, mapOperands, presburger::BoundType::EQ, val, ivDeps))) {
      LDBG() << "Value is not an affine function of the enclosing loop IVs: "
             << val;
      return failure();
    }
    operands = llvm::map_to_vector(
        mapOperands, [](const auto& vd) -> Value { return vd.first; });
  }

  return cst.addBound(type, pos, map, operands);
}

static LogicalResult addLoopDomain(affine::FlatAffineValueConstraints& cst,
                                   LoopLikeOpInterface loop,
                                   ArrayRef<Value> ivs,
                                   const ValueDimList& ivDeps) {
  std::optional<Value> iv = loop.getSingleInductionVar();
  std::optional<OpFoldResult> lbOfr = loop.getSingleLowerBound();
  std::optional<OpFoldResult> ubOfr = loop.getSingleUpperBound();
  std::optional<OpFoldResult> stepOfr = loop.getSingleStep();
  if (!iv || !lbOfr || !ubOfr || !stepOfr) return failure();

  std::optional<int64_t> step = getConstantIntValue(*stepOfr);
  if (!step || *step <= 0) return failure();

  unsigned pos;
  if (!cst.findVar(*iv, &pos)) return failure();

  // lb <= iv <= ub - 1
  if (failed(addAffineBound(cst, presburger::BoundType::LB, pos, *lbOfr, ivs,
                            ivDeps)) ||
      failed(addAffineBound(cst, presburger::BoundType::UB, pos, *ubOfr, ivs,
                            ivDeps))) {
    return failure();
  }

  if (*step != 1) {
    std::optional<int64_t> lb = getConstantIntValue(*lbOfr);
    if (!lb) return failure();
    SmallVector<int64_t> dividend(cst.getNumCols(), 0);
    dividend[pos] = 1;
    dividend.back() = -*lb;
    unsigned qPos = cst.addLocalFloorDiv(dividend, *step);

    SmallVector<int64_t> eq(cst.getNumCols(), 0);
    eq[pos] = 1;
    eq[qPos] = -*step;
    eq.back() = -*lb;
    cst.addEquality(eq);
  }
  return success();
}

static LogicalResult addAffineIfDomain(affine::FlatAffineValueConstraints& cst,
                                       affine::AffineIfOp ifOp, bool isElse,
                                       ArrayRef<Value> ivs,
                                       const ValueDimList& ivDeps) {
  IntegerSet set = ifOp.getIntegerSet();
  // If the rotation is in the `else` branch, negate the inequality
  if (isElse) {
    if (set.getNumConstraints() != 1 || set.isEq(0)) return failure();
    set = IntegerSet::get(set.getNumDims(), set.getNumSymbols(),
                          {-set.getConstraint(0) - 1}, /*eqFlags=*/{false});
  }
  SmallVector<Value> operands(ifOp.getOperands());
  affine::canonicalizeSetAndOperands(&set, &operands);

  FailureOr<affine::FlatAffineValueConstraints> ifCst =
      affine::FlatAffineValueConstraints::create(set, operands);
  if (failed(ifCst)) {
    LDBG() << "semi-affine integer sets are unsupported here";
    return failure();
  }

  ifCst->convertVarKind(presburger::VarKind::Symbol, 0,
                        ifCst->getNumSymbolVars(), presburger::VarKind::SetDim);

  unsigned prevNumDims = cst.getNumDimVars();
  cst.mergeAndAlignVarsWithOther(0, &*ifCst);
  cst.append(*ifCst);

  for (unsigned pos = prevNumDims; pos < cst.getNumDimVars(); ++pos) {
    if (failed(addAffineBound(cst, presburger::BoundType::EQ, pos,
                              cst.getValue(pos), ivs, ivDeps))) {
      return failure();
    }
  }
  return success();
}

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
  SmallVector<std::pair<affine::AffineIfOp, bool>> affineIfs;
  for (Operation* curr = op; curr->getParentOp(); curr = curr->getParentOp()) {
    Operation* parent = curr->getParentOp();
    if (isa<scf::IfOp>(parent)) return failure();
    if (auto loop = dyn_cast<LoopLikeOpInterface>(parent)) {
      loops.push_back(loop);
    } else if (auto affineIf = dyn_cast<affine::AffineIfOp>(parent)) {
      bool isElse = (curr->getParentRegion() == &affineIf.getElseRegion());
      affineIfs.push_back({affineIf, isElse});
    }
  }
  if (loops.empty()) return failure();

  SmallVector<Value, 4> ivs;
  for (LoopLikeOpInterface loop : loops) {
    std::optional<Value> iv = loop.getSingleInductionVar();
    if (!iv) return failure();
    ivs.push_back(*iv);
  }

  ValueDimList ivDeps = llvm::map_to_vector(ivs, [](Value iv) {
    return std::make_pair(iv, std::optional<int64_t>());
  });

  affine::FlatAffineValueConstraints cst(ivs.size(), /*numSymbols=*/0,
                                         /*numLocals=*/0, ivs);
  for (LoopLikeOpInterface loop : loops) {
    if (failed(addLoopDomain(cst, loop, ivs, ivDeps))) {
      LDBG() << "Failed to add loop domain constraints " << loop;
      return failure();
    }
  }

  for (const auto& [ifOp, isElse] : affineIfs) {
    if (failed(addAffineIfDomain(cst, ifOp, isElse, ivs, ivDeps))) {
      LDBG() << "Failed to add affine.if domain constraints " << ifOp;
      return failure();
    }
  }

  // Add constraint for each rotation index
  unsigned numDomainDims = cst.getNumDimVars();
  for (OpFoldResult ofr : rotationOp.getRotationIndices()) {
    unsigned shiftPos = cst.appendDimVar();
    if (failed(addAffineBound(cst, presburger::BoundType::EQ, shiftPos, ofr,
                              ivs, ivDeps))) {
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
