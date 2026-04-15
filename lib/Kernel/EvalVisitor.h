#ifndef LIB_KERNEL_EVALVISITOR_H_
#define LIB_KERNEL_EVALVISITOR_H_

#include <memory>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

// A visitor that evaluates an arithmetic DAG of ciphertext semantic tensors.
// The evaluation is done by replacing the leaves with their literal values and
// then computing the operations.
using EvalResults = std::vector<LiteralValue>;

class EvalVisitor : public CachingVisitor<LiteralValue, EvalResults> {
 public:
  using CachingVisitor<LiteralValue, EvalResults>::operator();

  EvalVisitor() : CachingVisitor<LiteralValue, EvalResults>() {}

  EvalResults operator()(const ConstantTensorNode& node) override;
  EvalResults operator()(const ConstantScalarNode& node) override;
  EvalResults operator()(const SplatNode& node) override;
  EvalResults operator()(const LeafNode<LiteralValue>& node) override;
  EvalResults operator()(const AddNode<LiteralValue>& node) override;
  EvalResults operator()(const SubtractNode<LiteralValue>& node) override;
  EvalResults operator()(const MultiplyNode<LiteralValue>& node) override;
  EvalResults operator()(const FloorDivNode<LiteralValue>& node) override;
  EvalResults operator()(const LeftRotateNode<LiteralValue>& node) override;
  EvalResults operator()(const ExtractNode<LiteralValue>& node) override;
  EvalResults operator()(const InsertNode<LiteralValue>& node) override;
  EvalResults operator()(const ComparisonNode<LiteralValue>& node) override;
  EvalResults operator()(const IfElseNode<LiteralValue>& node) override;
  EvalResults operator()(const VariableNode<LiteralValue>& node) override;
  EvalResults operator()(const ForLoopNode<LiteralValue>& node) override;
  EvalResults operator()(const YieldNode<LiteralValue>& node) override;
  EvalResults operator()(const ResultAtNode<LiteralValue>& node) override;
};

EvalResults evalKernel(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag);

std::vector<EvalResults> multiEvalKernel(
    ArrayRef<std::shared_ptr<ArithmeticDagNode<LiteralValue>>> dags);

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_EVALVISITOR_H_
