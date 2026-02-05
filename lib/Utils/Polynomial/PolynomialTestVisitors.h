#ifndef LIB_UTILS_POLYNOMIAL_POLYNOMIALTESTVISITORS_H_
#define LIB_UTILS_POLYNOMIAL_POLYNOMIALTESTVISITORS_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace polynomial {
namespace test {

using kernel::AddNode;
using kernel::CachingVisitor;
using kernel::ConstantScalarNode;
using kernel::ConstantTensorNode;
using kernel::SplatNode;
using kernel::ExtractNode;
using kernel::LeafNode;
using kernel::LeftRotateNode;
using kernel::LiteralDouble;
using kernel::MultiplyNode;
using kernel::PowerNode;
using kernel::SubtractNode;

// Visitor that evaluates an ArithmeticDag by performing actual arithmetic
// operations. Templated on the leaf node type T.
template <typename T>
class EvalVisitorImpl : public CachingVisitor<T, double> {
 public:
  using CachingVisitor<T, double>::operator();

  EvalVisitorImpl() : CachingVisitor<T, double>() {}

  double operator()(const ConstantScalarNode& node) override {
    return node.value;
  }

  double operator()(const LeafNode<T>& node) override {
    if constexpr (std::is_same_v<T, double>) {
      return node.value;
    } else if constexpr (std::is_same_v<T, LiteralDouble>) {
      return node.value.getValue();
    } else {
      assert(false && "Unsupported leaf node type");
      return 0.0;
    }
  }

  double operator()(const AddNode<T>& node) override {
    // Recursive calls use the public `process` method from the base class
    // to ensure caching is applied at every step.
    return this->process(node.left) + this->process(node.right);
  }

  double operator()(const SubtractNode<T>& node) override {
    return this->process(node.left) - this->process(node.right);
  }

  double operator()(const MultiplyNode<T>& node) override {
    return this->process(node.left) * this->process(node.right);
  }

  double operator()(const ConstantTensorNode& node) override {
    // Tensor nodes are not expected in scalar polynomial evaluation
    assert(false && "ConstantTensorNode not supported in scalar evaluation");
    return 0.0;
  }

  double operator()(const PowerNode<T>& node) override {
    double base = this->process(node.base);
    return std::pow(base, static_cast<double>(node.exponent));
  }

  double operator()(const LeftRotateNode<T>& node) override {
    // Rotation is a tensor operation, not expected in scalar evaluation
    assert(false && "LeftRotateNode not supported in scalar evaluation");
    return 0.0;
  }

  double operator()(const ExtractNode<T>& node) override {
    // Extraction is a tensor operation, not expected in scalar evaluation
    assert(false && "ExtractNode not supported in scalar evaluation");
    return 0.0;
  }
};

// Convenience alias for the most common use case
using EvalVisitor = EvalVisitorImpl<LiteralDouble>;

// Visitor that computes the multiplicative depth of an ArithmeticDag.
// Templated on the leaf node type T.
template <typename T>
class MultiplicativeDepthVisitorImpl : public CachingVisitor<T, double> {
 public:
  using CachingVisitor<T, double>::operator();

  MultiplicativeDepthVisitorImpl() : CachingVisitor<T, double>() {}

  double operator()(const ConstantScalarNode& node) override { return -1.0; }

  double operator()(const LeafNode<T>& node) override { return 0.0; }

  double operator()(const AddNode<T>& node) override {
    // Recursive calls use the public `process` method from the base class
    // to ensure caching is applied at every step.
    return std::max(this->process(node.left), this->process(node.right));
  }

  double operator()(const SubtractNode<T>& node) override {
    return std::max(this->process(node.left), this->process(node.right));
  }

  double operator()(const MultiplyNode<T>& node) override {
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

  double operator()(const PowerNode<T>& node) override {
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

  double operator()(const LeftRotateNode<T>& node) override {
    // Rotation doesn't increase multiplicative depth
    return this->process(node.operand);
  }

  double operator()(const ExtractNode<T>& node) override {
    // Extraction doesn't increase multiplicative depth
    return this->process(node.operand);
  }
};

// Convenience alias for the most common use case
using MultiplicativeDepthVisitor =
    MultiplicativeDepthVisitorImpl<LiteralDouble>;

}  // namespace test
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_POLYNOMIALTESTVISITORS_H_
