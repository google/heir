#include "lib/Utils/AffineLoopUtils.h"

#include <cstdint>
#include <optional>

#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"        // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"   // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"    // from @llvm-project
#include "mlir/include/mlir/Interfaces/ValueBoundsOpInterface.h"  // from @llvm-project

#define DEBUG_TYPE "affine-loop-utils"

namespace mlir {
namespace heir {

FailureOr<AffineExpr> getAffineExprOfIVs(Value value, ArrayRef<Value> ivs) {
  MLIRContext* ctx = value.getContext();
  const auto* it = llvm::find(ivs, value);
  if (it != ivs.end()) return getAffineDimExpr(it - ivs.begin(), ctx);

  APInt cst;
  if (matchPattern(value, m_ConstantInt(&cst)))
    return getAffineConstantExpr(cst.getSExtValue(), ctx);

  Operation* def = value.getDefiningOp();
  if (!def) return failure();

  auto getOperand = [&](unsigned i) {
    return getAffineExprOfIVs(def->getOperand(i), ivs);
  };
  auto getConstantRhs = [&]() -> std::optional<int64_t> {
    APInt rhs;
    if (!matchPattern(def->getOperand(1), m_ConstantInt(&rhs)))
      return std::nullopt;
    return rhs.getSExtValue();
  };

  return llvm::TypeSwitch<Operation*, FailureOr<AffineExpr>>(def)
      .Case<arith::AddIOp, arith::SubIOp, arith::MulIOp>(
          [&](auto op) -> FailureOr<AffineExpr> {
            FailureOr<AffineExpr> lhs = getOperand(0);
            FailureOr<AffineExpr> rhs = getOperand(1);
            if (failed(lhs) || failed(rhs)) return failure();
            if (isa<arith::AddIOp>(op)) return *lhs + *rhs;
            if (isa<arith::SubIOp>(op)) return *lhs - *rhs;
            // Only products with a constant are affine.
            if (!isa<AffineConstantExpr>(*lhs) &&
                !isa<AffineConstantExpr>(*rhs))
              return failure();
            return *lhs * *rhs;
          })
      .Case<arith::RemUIOp>([&](auto op) -> FailureOr<AffineExpr> {
        std::optional<int64_t> rhs = getConstantRhs();
        if (!rhs || *rhs <= 0 || !llvm::isPowerOf2_64(*rhs)) return failure();
        FailureOr<AffineExpr> lhs = getOperand(0);
        if (failed(lhs)) return failure();
        return *lhs % *rhs;
      })
      .Case<arith::FloorDivSIOp>([&](auto op) -> FailureOr<AffineExpr> {
        std::optional<int64_t> rhs = getConstantRhs();
        if (!rhs || *rhs <= 0) return failure();
        FailureOr<AffineExpr> lhs = getOperand(0);
        if (failed(lhs)) return failure();
        return lhs->floorDiv(*rhs);
      })
      .Case<arith::IndexCastOp, arith::IndexCastUIOp>(
          [&](auto op) { return getOperand(0); })
      .Case<affine::AffineApplyOp>(
          [&](affine::AffineApplyOp op) -> FailureOr<AffineExpr> {
            AffineMap map = op.getAffineMap();
            SmallVector<AffineExpr> dimReplacements;
            SmallVector<AffineExpr> symReplacements;
            for (auto [i, operand] : llvm::enumerate(op.getMapOperands())) {
              FailureOr<AffineExpr> expr = getAffineExprOfIVs(operand, ivs);
              if (failed(expr)) return failure();
              if (i < map.getNumDims())
                dimReplacements.push_back(*expr);
              else
                symReplacements.push_back(*expr);
            }
            return map.getResult(0).replaceDimsAndSymbols(dimReplacements,
                                                          symReplacements);
          })
      .Default([](Operation*) { return failure(); });
}

LogicalResult addAffineBound(affine::FlatAffineValueConstraints& cst,
                             presburger::BoundType type, unsigned pos,
                             OpFoldResult ofr, ArrayRef<Value> ivs) {
  if (std::optional<int64_t> c = getConstantIntValue(ofr)) {
    int64_t bound = (type == presburger::BoundType::UB) ? *c - 1 : *c;
    cst.addBound(type, pos, bound);
    return success();
  }

  Value val = cast<Value>(ofr);
  MLIRContext* ctx = val.getContext();
  AffineMap map;
  SmallVector<Value> operands;
  if (FailureOr<AffineExpr> expr = getAffineExprOfIVs(val, ivs);
      succeeded(expr)) {
    map = AffineMap::get(ivs.size(), /*symbolCount=*/0, *expr, ctx);
    operands.assign(ivs.begin(), ivs.end());
  } else {
    // Walk the use-def chain via ValueBoundsOpInterface and return an
    // AffineMap of the IVs.
    ValueDimList ivDeps = llvm::map_to_vector(ivs, [](Value iv) {
      return std::make_pair(iv, std::optional<int64_t>());
    });
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

LogicalResult addLoopDomain(affine::FlatAffineValueConstraints& cst,
                            LoopLikeOpInterface loop, ArrayRef<Value> ivs) {
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
  if (failed(
          addAffineBound(cst, presburger::BoundType::LB, pos, *lbOfr, ivs)) ||
      failed(
          addAffineBound(cst, presburger::BoundType::UB, pos, *ubOfr, ivs))) {
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

LogicalResult addAffineIfDomain(affine::FlatAffineValueConstraints& cst,
                                affine::AffineIfOp ifOp, bool isElse,
                                ArrayRef<Value> ivs) {
  IntegerSet set = ifOp.getIntegerSet();
  // If the op is in the `else` branch, negate the inequality
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
                              cst.getValue(pos), ivs))) {
      return failure();
    }
  }
  return success();
}

LogicalResult getEnclosingLoopsAndIfs(
    Operation* op, SmallVectorImpl<LoopLikeOpInterface>& loops,
    SmallVectorImpl<EnclosingAffineIf>& affineIfs) {
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
  return success();
}

FailureOr<affine::FlatAffineValueConstraints> getIterationDomain(
    ArrayRef<LoopLikeOpInterface> loops,
    ArrayRef<EnclosingAffineIf> affineIfs) {
  SmallVector<Value, 4> ivs;
  for (LoopLikeOpInterface loop : loops) {
    std::optional<Value> iv = loop.getSingleInductionVar();
    if (!iv) return failure();
    ivs.push_back(*iv);
  }

  affine::FlatAffineValueConstraints cst(ivs.size(), /*numSymbols=*/0,
                                         /*numLocals=*/0, ivs);
  for (LoopLikeOpInterface loop : loops) {
    if (failed(addLoopDomain(cst, loop, ivs))) {
      LDBG() << "Failed to add loop domain constraints " << loop;
      return failure();
    }
  }

  for (const auto& [ifOp, isElse] : affineIfs) {
    if (failed(addAffineIfDomain(cst, ifOp, isElse, ivs))) {
      LDBG() << "Failed to add affine.if domain constraints " << ifOp;
      return failure();
    }
  }
  return cst;
}

}  // namespace heir
}  // namespace mlir
