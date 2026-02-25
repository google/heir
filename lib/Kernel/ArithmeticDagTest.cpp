#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

using StringLeavedDag = ArithmeticDagNode<std::string>;
using DoubleLeavedDag = ArithmeticDagNode<double>;

struct FlattenedStringVisitor {
  std::string operator()(const ConstantScalarNode& node) const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << node.value;
    return ss.str();
  }

  std::string operator()(const ConstantTensorNode& node) const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < node.value.size(); ++i) {
      ss << node.value[i];
      if (i != node.value.size() - 1) {
        ss << ", ";
      }
    }
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

  std::string operator()(const FloorDivNode<std::string>& node) const {
    std::stringstream ss;
    ss << "(" << node.left->visit(*this) << " / "
       << std::to_string(node.divisor) << ")";
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

  std::string operator()(const SplatNode& node) const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "splat(" << node.value << ")";
    return ss.str();
  }
};

class EvalVisitor : public CachingVisitor<double, std::vector<double>> {
 public:
  // This is required for this class to see all overloads of the visit function,
  // including virtual ones not implemented by this class.
  using CachingVisitor<double, std::vector<double>>::operator();

  EvalVisitor() : CachingVisitor<double, std::vector<double>>(), callCount(0) {}

  // To test that caching works as expected.
  int callCount;
  std::vector<double> operator()(const ConstantScalarNode& node) override {
    callCount += 1;
    return {node.value};
  }

  std::vector<double> operator()(const LeafNode<double>& node) override {
    callCount += 1;
    return {node.value};
  }

  std::vector<double> operator()(const AddNode<double>& node) override {
    // Recursive calls use the public `process` method from the base class
    // to ensure caching is applied at every step.
    callCount += 1;
    return {this->process(node.left)[0] + this->process(node.right)[0]};
  }

  std::vector<double> operator()(const SubtractNode<double>& node) override {
    callCount += 1;
    return {this->process(node.left)[0] - this->process(node.right)[0]};
  }
  std::vector<double> operator()(const MultiplyNode<double>& node) override {
    callCount += 1;
    return {this->process(node.left)[0] * this->process(node.right)[0]};
  }

  std::vector<double> operator()(const PowerNode<double>& node) override {
    callCount += 1;
    return {std::pow(this->process(node.base)[0], node.exponent)};
  }

  std::vector<double> operator()(const SplatNode& node) override {
    callCount += 1;
    return {node.value};
  }
};

TEST(ArithmeticDagTest, TestPrint) {
  auto root = StringLeavedDag::leftRotate(
      StringLeavedDag::mul(
          StringLeavedDag::add(
              StringLeavedDag::leaf("x"),
              StringLeavedDag::constantScalar(3.0, DagType::floatTy(64))),
          StringLeavedDag::power(StringLeavedDag::leaf("y"), 2)),
      7);

  FlattenedStringVisitor visitor;
  std::string result = root->visit(visitor);
  EXPECT_EQ(result, "(((x + 3.00) * (y ^ 2)) << 7)");
}

TEST(ArithmeticDagTest, TestDiv) {
  auto root = StringLeavedDag::leftRotate(
      StringLeavedDag::mul(
          StringLeavedDag::floorDiv(StringLeavedDag::leaf("x"), 3),
          StringLeavedDag::power(StringLeavedDag::leaf("y"), 2)),
      7);

  FlattenedStringVisitor visitor;
  std::string result = root->visit(visitor);
  EXPECT_EQ(result, "(((x / 3) * (y ^ 2)) << 7)");
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
  auto root = DoubleLeavedDag::mul(
      DoubleLeavedDag::add(shared, shared),
      DoubleLeavedDag::constantScalar(3.0, DagType::floatTy(64)));

  EvalVisitor visitor;
  double result = root->visit(visitor)[0];
  EXPECT_EQ(result, 24.0);
  EXPECT_EQ(visitor.callCount, 5);
}

TEST(ArithmeticDagTest, TestEvaluationVisitorSubstract) {
  auto x = DoubleLeavedDag::leaf(1.0);
  auto y = DoubleLeavedDag::leaf(2.0);
  auto root = DoubleLeavedDag::sub(x, y);

  EvalVisitor visitor;
  double result = root->visit(visitor)[0];
  EXPECT_EQ(result, -1);
  EXPECT_EQ(visitor.callCount, 3);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
