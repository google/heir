#ifndef LIB_ANALYSIS_ROTATIONANALYSIS_ROTATIONEVALVISITOR_H_
#define LIB_ANALYSIS_ROTATIONANALYSIS_ROTATIONEVALVISITOR_H_

#include <memory>
#include <set>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"

namespace mlir {
namespace heir {

class RotationEvalVisitor : public kernel::EvalVisitor {
 public:
  using EvalVisitor::operator();
  using EvalVisitor::EvalVisitor;

  // Override the rotation op to record the materialized rotation shift.
  kernel::EvalResults operator()(
      const kernel::LeftRotateNode<kernel::LiteralValue>& node) override;

  // Override the bulk rotation op to record all materialized rotation shifts.
  kernel::EvalResults operator()(
      const kernel::LeftRotateBulkNode<kernel::LiteralValue>& node) override;

  // Override the insert node to keep the IR connected without actual
  // computation.
  kernel::EvalResults operator()(
      const kernel::InsertNode<kernel::LiteralValue>& node) override;

  // Override the variable node to allow uninitialized values to be populated
  // with anything.
  kernel::EvalResults operator()(
      const kernel::VariableNode<kernel::LiteralValue>& node) override;

  const std::set<int>& getEvaluatedShifts() const { return evaluatedShifts; }

 private:
  std::set<int> evaluatedShifts;
};

std::set<int> evalRotations(
    const std::shared_ptr<kernel::ArithmeticDagNode<kernel::LiteralValue>>&
        dag);

}  // namespace heir
}  // namespace mlir
#endif  // LIB_ANALYSIS_ROTATIONANALYSIS_ROTATIONEVALVISITOR_H_
