#include <cassert>
#include <cmath>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace {

using StringLeavedDag = ArithmeticDagNode<std::string>;
using DoubleLeavedDag = ArithmeticDagNode<double>;

struct FlattenedStringVisitor {
  std::string operator()(const ConstantNode& node) const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << node.value;
    return ss.str();
  }

  std::string operator()(const LeafNode<std::string>& node) const {
    return node.value;
  }

  std::string operator()(const AddNode<std::string>& node) const {
    std::stringstream ss;
    ss << "(" << node.left->visit(*this) << " + " << node.right->visit(*this)
       << ")";
    return ss.str();
  }

  std::string operator()(const SubtractNode<std::string>& node) const {
    std::stringstream ss;
    ss << "(" << node.left->visit(*this) << " - " << node.right->visit(*this)
       << ")";
    return ss.str();
  }

  std::string operator()(const MultiplyNode<std::string>& node) const {
    std::stringstream ss;
    ss << "(" << node.left->visit(*this) << " * " << node.right->visit(*this)
       << ")";
    return ss.str();
  }

  std::string operator()(const PowerNode<std::string>& node) const {
    std::stringstream ss;
    ss << "(" << node.base->visit(*this) << " ^ " << node.exponent << ")";
    return ss.str();
  }

  std::string operator()(const LeftRotateNode<std::string>& node) const {
    std::stringstream ss;
    ss << "(" << node.operand->visit(*this) << " << " << node.shift << ")";
    return ss.str();
  }

  std::string operator()(const ExtractNode<std::string>& node) const {
    std::stringstream ss;
    ss << node.operand->visit(*this) << "[" << node.index << "]";
    return ss.str();
  }
};

class EvalVisitor : public CachingVisitor<double, double> {
 public:
  // This is required for this class to see all overloads of the visit function,
  // including virtual ones not implemented by this class.
  using CachingVisitor<double, double>::operator();

  EvalVisitor() : CachingVisitor<double, double>(), callCount(0) {}

  // To test that caching works as expected.
  int callCount;
  double operator()(const ConstantNode& node) override {
    callCount += 1;
    return node.value;
  }

  double operator()(const LeafNode<double>& node) override {
    callCount += 1;
    return node.value;
  }

  double operator()(const AddNode<double>& node) override {
    // Recursive calls use the public `process` method from the base class
    // to ensure caching is applied at every step.
    callCount += 1;
    return this->process(node.left) + this->process(node.right);
  }

  double operator()(const SubtractNode<double>& node) override {
    callCount += 1;
    return this->process(node.left) - this->process(node.right);
  }
  double operator()(const MultiplyNode<double>& node) override {
    callCount += 1;
    return this->process(node.left) * this->process(node.right);
  }

  double operator()(const PowerNode<double>& node) override {
    callCount += 1;
    return std::pow(this->process(node.base), node.exponent);
  }
};

TEST(ArithmeticDagTest, TestPrint) {
  auto root = StringLeavedDag::leftRotate(
      StringLeavedDag::mul(
          StringLeavedDag::add(StringLeavedDag::leaf("x"),
                               StringLeavedDag::constant(3.0)),
          StringLeavedDag::power(StringLeavedDag::leaf("y"), 2)),
      7);

  FlattenedStringVisitor visitor;
  std::string result = root->visit(visitor);
  EXPECT_EQ(result, "(((x + 3.00) * (y ^ 2)) << 7)");
}

TEST(ArithmeticDagTest, TestProperDag) {
  auto shared = StringLeavedDag::power(StringLeavedDag::leaf("y"), 2);
  auto root =
      StringLeavedDag::mul(StringLeavedDag::add(shared, shared), shared);

  FlattenedStringVisitor visitor;
  std::string result = root->visit(visitor);
  EXPECT_EQ(result, "(((y ^ 2) + (y ^ 2)) * (y ^ 2))");
}

TEST(ArithmeticDagTest, TestEvaluationVisitor) {
  auto shared = DoubleLeavedDag::power(DoubleLeavedDag::leaf(2.0), 2);
  auto root = DoubleLeavedDag::mul(DoubleLeavedDag::add(shared, shared),
                                   DoubleLeavedDag::constant(3.0));

  EvalVisitor visitor;
  double result = root->visit(visitor);
  EXPECT_EQ(result, 24.0);
  EXPECT_EQ(visitor.callCount, 5);
}

TEST(ArithmeticDagTest, TestEvaluationVisitorSubstract) {
  auto x = DoubleLeavedDag::leaf(1.0);
  auto y = DoubleLeavedDag::leaf(2.0);
  auto root = DoubleLeavedDag::sub(x, y);

  EvalVisitor visitor;
  double result = root->visit(visitor);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(visitor.callCount, 3);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
