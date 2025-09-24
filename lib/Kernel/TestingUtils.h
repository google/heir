#ifndef LIB_KERNEL_TESTINGUTILS_H_
#define LIB_KERNEL_TESTINGUTILS_H_

#include <memory>
#include <string>
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
class EvalVisitor : public CachingVisitor<LiteralValue, LiteralValue> {
 public:
  using CachingVisitor<LiteralValue, LiteralValue>::operator();

  EvalVisitor() : CachingVisitor<LiteralValue, LiteralValue>() {}

  LiteralValue operator()(const ConstantTensorNode& node) override;
  LiteralValue operator()(const LeafNode<LiteralValue>& node) override;
  LiteralValue operator()(const AddNode<LiteralValue>& node) override;
  LiteralValue operator()(const SubtractNode<LiteralValue>& node) override;
  LiteralValue operator()(const MultiplyNode<LiteralValue>& node) override;
  LiteralValue operator()(const LeftRotateNode<LiteralValue>& node) override;
  LiteralValue operator()(const ExtractNode<LiteralValue>& node) override;
};

LiteralValue evalKernel(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag);

std::vector<LiteralValue> multiEvalKernel(
    ArrayRef<std::shared_ptr<ArithmeticDagNode<LiteralValue>>> dags);

// A visitor that prints the dag in textual form
class PrintVisitor : public CachingVisitor<LiteralValue, std::string> {
 public:
  using CachingVisitor<LiteralValue, std::string>::operator();

  PrintVisitor() : CachingVisitor<LiteralValue, std::string>() {}

  std::string operator()(const LeafNode<LiteralValue>& node) override;
  std::string operator()(const AddNode<LiteralValue>& node) override;
  std::string operator()(const SubtractNode<LiteralValue>& node) override;
  std::string operator()(const MultiplyNode<LiteralValue>& node) override;
  std::string operator()(const LeftRotateNode<LiteralValue>& node) override;
  std::string operator()(const ExtractNode<LiteralValue>& node) override;
};

std::string printKernel(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag);

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_TESTINGUTILS_H_
