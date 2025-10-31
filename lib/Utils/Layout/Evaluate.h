#ifndef LIB_UTILS_LAYOUT_EVALUATE_H_
#define LIB_UTILS_LAYOUT_EVALUATE_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project

// ISL
#include "include/isl/ctx.h"         // from @isl
#include "include/isl/map.h"         // from @isl
#include "include/isl/map_type.h"    // from @isl
#include "include/isl/point.h"       // from @isl
#include "include/isl/set.h"         // from @isl
#include "include/isl/space_type.h"  // from @isl
#include "include/isl/val.h"         // from @isl
#include "include/isl/val_type.h"    // from @isl

namespace mlir {
namespace heir {

using presburger::BoundType;
using presburger::VarKind;

template <typename T>
std::vector<std::vector<T>> evaluateLayout(
    const presburger::IntegerRelation& relation,
    std::function<T(const std::vector<int64_t>&)> getValueFn) {
  auto numCt = relation.getConstantBound64(
      BoundType::UB, relation.getVarKindOffset(VarKind::Range));
  assert(numCt.has_value() &&
         "expected a constant bound for the number of ciphertexts");
  auto numSlots = relation.getConstantBound64(
      BoundType::UB, relation.getVarKindOffset(VarKind::Range) + 1);
  assert(numSlots.has_value() &&
         "expected a constant bound for the number of slots");

  std::vector<std::vector<T>> result(numCt.value() + 1,
                                     std::vector<T>(numSlots.value() + 1, 0));

  // Get all points in the relation.
  PointPairCollector collector(relation.getNumDomainVars(), /*rangeDims=*/2);
  enumeratePoints(relation, collector);

  for (const auto& pointPair : collector.points) {
    std::vector<int64_t> domainPoint = pointPair.first;
    std::vector<int64_t> rangePoint = pointPair.second;
    result[rangePoint[0]][rangePoint[1]] = getValueFn(domainPoint);
  }
  return result;
}

// This applies the layout relation on a given input vector.
//
// The layout relation is a presburger relation with domain size equal to the
// rank of the input vector and range size equal to two (ct, slot). The return
// value is a 2-D vector whose ct, slot entries consist of the data that is
// mapped from the layout relation.

// This is intended to be used for test vectors, since the performance is not
// optimized.
template <typename T>
std::vector<std::vector<T>> evaluateLayoutOnVector(
    const presburger::IntegerRelation& relation, const std::vector<T>& input) {
  std::function<T(const std::vector<int64_t>&)> getValueFn =
      [&](const std::vector<int64_t>& domainPoint) {
        return input[domainPoint[0]];
      };
  return evaluateLayout(relation, getValueFn);
}

// This applies the layout relation on a given input 2-D vector. See above.

template <typename T>
std::vector<std::vector<T>> evaluateLayoutOnMatrix(
    const presburger::IntegerRelation& relation,
    const std::vector<std::vector<T>>& input) {
  std::function<T(const std::vector<int64_t>&)> getValueFn =
      [&](const std::vector<int64_t>& domainPoint) {
        return input[domainPoint[0]][domainPoint[1]];
      };
  return evaluateLayout(relation, getValueFn);
}

template <typename T>
std::vector<std::vector<T>> unpackLayoutToMatrix(
    const presburger::IntegerRelation& relation,
    const std::vector<std::vector<T>>& packed,
    ArrayRef<int64_t> originalShape) {
  std::vector<std::vector<T>> result(originalShape[0],
                                     std::vector<T>(originalShape[1], 0));

  // Get all points in the relation.
  PointPairCollector collector(relation.getNumDomainVars(), /*rangeDims=*/2);
  enumeratePoints(relation, collector);

  for (const auto& pointPair : collector.points) {
    std::vector<int64_t> domainPoint = pointPair.first;
    std::vector<int64_t> rangePoint = pointPair.second;
    result[domainPoint[0]][domainPoint[1]] =
        packed[rangePoint[0]][rangePoint[1]];
  }
  return result;
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_EVALUATE_H_
