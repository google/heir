#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

using StringLeavedDag = ArithmeticDagNode<std::string>;
using DoubleLeavedDag = ArithmeticDagNode<double>;

struct FlattenedStringVisitor {
  mutable std::unordered_map<const VariableNode<std::string>*, std::string> varNames;
  mutable int varCounter = 0;

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

  std::string operator()(const VariableNode<std::string>& node) const {
    if (varNames.find(&node) == varNames.end()) {
      varNames[&node] = "i" + std::to_string(varCounter++);
    }
    return varNames[&node];
  }

  std::string operator()(const ForLoopNode<std::string>& node) const {
    std::stringstream ss;
    std::string inductionVarName = node.inductionVar->visit(*this);
    std::string iterArgName = node.iterArg->visit(*this);
    ss << "for(" << inductionVarName << "=" << node.lower << " to " << node.upper
       << " step " << node.step << "; " << iterArgName << "="
       << node.init->visit(*this) << ") { ";
    if (node.body) {
      ss << node.body->visit(*this);
    }
    ss << " }";
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
  double operator()(const ConstantScalarNode& node) override {
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

  double operator()(const VariableNode<double>& node) override {
    callCount += 1;
    assert(node.value.has_value() && "VariableNode must have a value during evaluation");
    return node.value.value();
  }

  void clearSubtreeCache(const std::shared_ptr<ArithmeticDagNode<double>>& node) {
    if (!node) return;

    clearCacheEntry(node.get());

    // Recursively clear cache for child nodes
    std::visit([this](auto&& n) {
      using NodeType = std::decay_t<decltype(n)>;
      if constexpr (std::is_same_v<NodeType, AddNode<double>> ||
                    std::is_same_v<NodeType, SubtractNode<double>> ||
                    std::is_same_v<NodeType, MultiplyNode<double>>) {
        clearSubtreeCache(n.left);
        clearSubtreeCache(n.right);
      } else if constexpr (std::is_same_v<NodeType, PowerNode<double>> ||
                           std::is_same_v<NodeType, LeftRotateNode<double>> ||
                           std::is_same_v<NodeType, ExtractNode<double>>) {
        if constexpr (std::is_same_v<NodeType, PowerNode<double>>) {
          clearSubtreeCache(n.base);
        } else {
          clearSubtreeCache(n.operand);
        }
      } else if constexpr (std::is_same_v<NodeType, ForLoopNode<double>>) {
        clearSubtreeCache(n.init);
        clearSubtreeCache(n.inductionVar);
        clearSubtreeCache(n.iterArg);
        clearSubtreeCache(n.body);
      }
    }, node->node_variant);
  }

  double operator()(const ForLoopNode<double>& node) override {
    callCount += 1;
    double result = this->process(node.init);
    for (size_t i = node.lower; i < node.upper; i += node.step) {
      // Set the induction variable value
      auto& inductionVarNode = std::get<VariableNode<double>>(node.inductionVar->node_variant);
      inductionVarNode.value = static_cast<double>(i);

      // Set the iter_arg value
      auto& iterArgNode = std::get<VariableNode<double>>(node.iterArg->node_variant);
      iterArgNode.value = result;

      // Clear the cache for the body and its dependencies before each iteration
      // since the body depends on variables that change each iteration
      if (node.body) {
        clearSubtreeCache(node.body);
        result = this->process(node.body);
      }
    }
    return result;
  }
};

TEST(ArithmeticDagTest, TestPrint) {
  auto root = StringLeavedDag::leftRotate(
      StringLeavedDag::mul(
          StringLeavedDag::add(StringLeavedDag::leaf("x"),
                               StringLeavedDag::constantScalar(3.0)),
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
                                   DoubleLeavedDag::constantScalar(3.0));

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

TEST(ArithmeticDagTest, TestLoop) {
  auto x = DoubleLeavedDag::leaf(1.0);
  auto loop = DoubleLeavedDag::loop(x, 0, 10, 1);
  auto& loopNode = std::get<ForLoopNode<double>>(loop->node_variant);
  loopNode.body = DoubleLeavedDag::add(x, loopNode.iterArg);
  EvalVisitor visitor;
  double result = loop->visit(visitor);
  EXPECT_EQ(result, 11);
}

TEST(ArithmeticDagTest, TestLoopStringVisitor) {
  auto x = StringLeavedDag::leaf("x");
  auto loop = StringLeavedDag::loop(x, 0, 10, 1);
  auto& loopNode = std::get<ForLoopNode<std::string>>(loop->node_variant);
  loopNode.body = StringLeavedDag::add(
      StringLeavedDag::mul(loopNode.inductionVar, StringLeavedDag::leaf("y")),
      loopNode.iterArg);

  FlattenedStringVisitor visitor;
  std::string result = loop->visit(visitor);
  EXPECT_EQ(result, "for(i0=0 to 10 step 1; i1=x) { ((i0 * y) + i1) }");
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
