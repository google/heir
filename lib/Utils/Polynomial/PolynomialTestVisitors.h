#ifndef LIB_UTILS_POLYNOMIAL_POLYNOMIALTESTVISITORS_H_
#define LIB_UTILS_POLYNOMIAL_POLYNOMIALTESTVISITORS_H_

#include <algorithm>

#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace polynomial {
namespace test {

using kernel::AddNode;
using kernel::CachingVisitor;
using kernel::ConstantNode;
using kernel::LeafNode;
using kernel::MultiplyNode;
using kernel::SubtractNode;

// Visitor that evaluates an ArithmeticDag by performing actual arithmetic
// operations
class EvalVisitor : public CachingVisitor<double, double> {
 public:
  using CachingVisitor<double, double>::operator();

  EvalVisitor() : CachingVisitor<double, double>() {}

  double operator()(const ConstantNode& node) override { return node.value; }

  double operator()(const LeafNode<double>& node) override {
    return node.value;
  }

  double operator()(const AddNode<double>& node) override {
    // Recursive calls use the public `process` method from the base class
    // to ensure caching is applied at every step.
    return this->process(node.left) + this->process(node.right);
  }

  double operator()(const SubtractNode<double>& node) override {
    return this->process(node.left) - this->process(node.right);
  }

  double operator()(const MultiplyNode<double>& node) override {
    return this->process(node.left) * this->process(node.right);
  }
};

// Visitor that computes the multiplicative depth of an ArithmeticDag
class MultiplicativeDepthVisitor : public CachingVisitor<double, double> {
 public:
  using CachingVisitor<double, double>::operator();

  MultiplicativeDepthVisitor() : CachingVisitor<double, double>() {}

  double operator()(const ConstantNode& node) override { return -1.0; }

  double operator()(const LeafNode<double>& node) override { return 0.0; }

  double operator()(const AddNode<double>& node) override {
    // Recursive calls use the public `process` method from the base class
    // to ensure caching is applied at every step.
    return std::max(this->process(node.left), this->process(node.right));
  }

  double operator()(const SubtractNode<double>& node) override {
    return std::max(this->process(node.left), this->process(node.right));
  }

  double operator()(const MultiplyNode<double>& node) override {
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
};

}  // namespace test
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_POLYNOMIALTESTVISITORS_H_
