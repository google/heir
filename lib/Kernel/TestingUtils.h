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

// Visitor that computes the multiplicative depth of an ArithmeticDag.
// Templated on the leaf node type T.
class MultiplicativeDepthVisitorImpl
    : public CachingVisitor<LiteralValue, double> {
 public:
  using CachingVisitor<LiteralValue, double>::operator();

  MultiplicativeDepthVisitorImpl() : CachingVisitor<LiteralValue, double>() {}

  double operator()(const ConstantScalarNode& node) override { return -1.0; }

  double operator()(const LeafNode<LiteralValue>& node) override { return 0.0; }

  double operator()(const AddNode<LiteralValue>& node) override {
    // Recursive calls use the public `process` method from the base class
    // to ensure caching is applied at every step.
    return std::max(this->process(node.left), this->process(node.right));
  }

  double operator()(const SubtractNode<LiteralValue>& node) override {
    return std::max(this->process(node.left), this->process(node.right));
  }

  double operator()(const MultiplyNode<LiteralValue>& node) override {
    double left = this->process(node.left);
    double right = this->process(node.right);
    if (left < 0.0) {
      return right;
    }
    if (right < 0.0) {
      return left;
    }
    return std::max(left, right) + 1.0;
  }

  double operator()(const ConstantTensorNode& node) override {
    // Constant tensors have no multiplicative depth
    return -1.0;
  }

  double operator()(const PowerNode<LiteralValue>& node) override {
    double base_depth = this->process(node.base);
    if (base_depth < 0.0) {
      // Base is a constant, so the power is also constant
      return -1.0;
    }
    if (node.exponent == 0) {
      return -1.0;  // x^0 = 1 (constant)
    }
    if (node.exponent == 1) {
      return base_depth;  // x^1 = x
    }
    // For exponent > 1, compute depth using repeated squaring approach
    // Number of multiplications needed is ceil(log2(exponent))
    size_t exp = node.exponent;
    int mult_count = 0;
    while (exp > 1) {
      exp = exp / 2;
      mult_count++;
    }
    return base_depth + static_cast<double>(mult_count);
  }

  double operator()(const LeftRotateNode<LiteralValue>& node) override {
    // Rotation doesn't increase multiplicative depth
    return this->process(node.operand);
  }

  double operator()(const ExtractNode<LiteralValue>& node) override {
    // Extraction doesn't increase multiplicative depth
    return this->process(node.operand);
  }
};

double evalMultiplicativeDepth(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag);

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_TESTINGUTILS_H_
