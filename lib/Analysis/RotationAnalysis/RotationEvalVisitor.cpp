#include "lib/Analysis/RotationAnalysis/RotationEvalVisitor.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <set>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project

#define DEBUG_TYPE "rotation-analysis"

namespace mlir {
namespace heir {

using kernel::ArithmeticDagNode;
using kernel::DagType;
using kernel::EvalResults;
using kernel::LeftRotateNode;
using kernel::LiteralValue;
using kernel::VariableNode;

// This is a copy of EvalVisitor::operator() for LeftRotateNode, but recording
// the materialized rotation amount.
EvalResults RotationEvalVisitor::operator()(
    const LeftRotateNode<LiteralValue>& node) {
  auto operand = this->process(node.operand)[0];
  auto shape = operand.getShape();
  assert(!shape.empty() && "rotate operand must be a tensor");
  auto dim = shape[0];

  auto evaluatedShift = this->process(node.shift)[0];
  int amount = std::get<int>(evaluatedShift.get());
  // Normalize amount to be in [0, dim)
  amount = ((amount % dim) + dim) % dim;

  // Save the evaluated shift
  evaluatedShifts.insert(amount);

  const auto& oVal = operand.get();
  const auto* oVec = std::get_if<std::vector<int>>(&oVal);
  assert(oVec && "unsupported rotate operand type");

  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*oVec)[(i + amount) % oVec->size()];
  }
  return {result};
}

EvalResults RotationEvalVisitor::operator()(
    const VariableNode<LiteralValue>& node) {
  if (node.value.has_value()) {
    return {node.value.value()};
  }

  // If the variable is not set, then it came from an SSA value that was not
  // directly part of the defined kernel. We populate the variable with a
  // zero-valued tensor because the actual values don't matter, just the
  // rotation indices.
  DagType dagType = node.type;
  return std::visit(
      [&](auto&& arg) -> EvalResults {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, kernel::IntegerType>) {
          return {LiteralValue(0)};
        } else if constexpr (std::is_same_v<T, kernel::FloatType>) {
          return {LiteralValue(0)};
        } else if constexpr (std::is_same_v<T, kernel::IndexType>) {
          return {LiteralValue(0)};
        } else if constexpr (std::is_same_v<T, kernel::IntTensorType>) {
          auto shape = arg.shape;
          assert(!shape.empty() && "expected nonempty shape");
          if (shape.size() == 1) {
            return {LiteralValue(std::vector<int>(shape[0], 0))};
          }
          return {LiteralValue(std::vector<std::vector<int>>(
              shape[0], std::vector<int>(shape[1], 0)))};
        } else if constexpr (std::is_same_v<T, kernel::FloatTensorType>) {
          auto shape = arg.shape;
          assert(!shape.empty() && "expected nonempty shape");
          if (shape.size() == 1) {
            return {LiteralValue(std::vector<int>(shape[0], 0))};
          }
          return {LiteralValue(std::vector<std::vector<int>>(
              shape[0], std::vector<int>(shape[1], 0)))};
        }
        llvm_unreachable("Unknown DagType variant");
      },
      dagType.type_variant);
}

std::set<int> evalRotations(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag) {
  RotationEvalVisitor visitor;
  auto evalResults = visitor.process(dag);
  return std::move(visitor.getEvaluatedShifts());
}

}  // namespace heir
}  // namespace mlir
