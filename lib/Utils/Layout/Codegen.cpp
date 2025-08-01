#include "lib/Utils/Layout/Codegen.h"

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"  // from @llvm-project

namespace mlir {
namespace heir {

using presburger::BoundType;
using presburger::IntegerRelation;

FailureOr<std::pair<int64_t, int64_t>> getProjectedBounds(
    const IntegerRelation& rel, int varIndex) {
  std::unique_ptr<IntegerRelation> cloneRel = rel.clone();
  cloneRel->projectOut(0, varIndex);
  cloneRel->projectOut(1, cloneRel->getNumVars() - 1);
  std::optional<int64_t> lower = cloneRel->getConstantBound64(BoundType::LB, 0);
  std::optional<int64_t> upper = cloneRel->getConstantBound64(BoundType::UB, 0);

  if (!lower.has_value() || !upper.has_value()) {
    return failure();
  }
  return std::make_pair(lower.value(), upper.value());
}

FailureOr<LoopNest> generateLoopNest(const IntegerRelation& rel,
                                     MLIRContext* context) {
  LoopNest nest;

  // Generate a naive loop nest with no simplification
  for (int i = 0; i < rel.getNumVars(); ++i) {
    auto res = getProjectedBounds(rel, i);
    if (failed(res)) {
      return failure();
    }
    auto [lower, upper] = res.value();
    nest.lowerBounds.push_back(lower);
    nest.upperBounds.push_back(upper);
    nest.numIterationVars++;
  }

  for (int i = 0; i < rel.getNumEqualities(); ++i) {
    auto constraint = rel.getEquality(i);
    // FIXME: convert to AffineExpr
    nest.conditions.push_back(constraint.getAffineExpr());
  }

  for (int i = 0; i < rel.getNumInequalities(); ++i) {
    auto constraint = rel.getInequality(i);
    // FIXME: convert to AffineExpr
    nest.conditions.push_back(constraint.getAffineExpr());
  }
  return nest;
}

}  // namespace heir
}  // namespace mlir
