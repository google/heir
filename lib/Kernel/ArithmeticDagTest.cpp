#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

using StringLeavedDag = ArithmeticDagNode<std::string>;
using DoubleLeavedDag = ArithmeticDagNode<double>;

struct FlattenedStringVisitor {
  mutable std::unordered_map<const VariableNode<std::string>*, std::string>
      varNames;
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
    ss << "(" << node.operand->visit(*this) << " << "
       << node.shift->visit(*this) << ")";
    return ss.str();
  }

  std::string operator()(const ExtractNode<std::string>& node) const {
    std::stringstream ss;
    ss << node.operand->visit(*this) << "[" << node.index->visit(*this) << "]";
    return ss.str();
  }

  std::string operator()(const ComparisonNode<std::string>& node) const {
    std::stringstream ss;
    std::string op;
    switch (node.predicate) {
      case ComparisonPredicate::LT:
        op = "<";
        break;
      case ComparisonPredicate::LE:
        op = "<=";
        break;
      case ComparisonPredicate::GT:
        op = ">";
        break;
      case ComparisonPredicate::GE:
        op = ">=";
        break;
      case ComparisonPredicate::EQ:
        op = "==";
        break;
      case ComparisonPredicate::NE:
        op = "!=";
        break;
    }
    ss << "(" << node.left->visit(*this) << " " << op << " "
       << node.right->visit(*this) << ")";
    return ss.str();
  }

  std::string operator()(const IfElseNode<std::string>& node) const {
    std::stringstream ss;
    ss << "if (" << node.condition->visit(*this) << ") { "
       << node.thenBody->visit(*this) << " } else { "
       << node.elseBody->visit(*this) << " }";
    return ss.str();
  }

  std::string operator()(const SplatNode& node) const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "splat(" << node.value << ")";
    return ss.str();
  }

  std::string operator()(const VariableNode<std::string>& node) const {
    if (varNames.find(&node) == varNames.end()) {
      varNames[&node] = "i" + std::to_string(varCounter++);
    }
    return varNames[&node];
  }

  std::string operator()(const YieldNode<std::string>& node) const {
    // For a single element, don't wrap in parentheses
    if (node.elements.size() == 1) {
      return node.elements[0]->visit(*this);
    }
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < node.elements.size(); ++i) {
      ss << node.elements[i]->visit(*this);
      if (i != node.elements.size() - 1) {
        ss << ", ";
      }
    }
    ss << ")";
    return ss.str();
  }

  std::string operator()(const ResultAtNode<std::string>& node) const {
    std::stringstream ss;
    ss << node.operand->visit(*this) << "#" << node.index;
    return ss.str();
  }

  std::string operator()(const ForLoopNode<std::string>& node) const {
    std::stringstream ss;
    std::string inductionVarName = node.inductionVar->visit(*this);
    std::vector<std::string> visitedIterArgs;
    std::vector<std::string> visitedInits;
    for (size_t i = 0; i < node.inits.size(); ++i) {
      visitedIterArgs.push_back(node.iterArgs[i]->visit(*this));
      visitedInits.push_back(node.inits[i]->visit(*this));
    }
    std::string initsString;
    for (size_t i = 0; i < visitedIterArgs.size(); ++i) {
      initsString += visitedIterArgs[i] + "=" + visitedInits[i];
      if (i != visitedIterArgs.size() - 1) {
        initsString += ", ";
      }
    }
    ss << "for(" << inductionVarName << "=" << node.lower << " to "
       << node.upper << " step " << node.step << "; " << initsString << ") { "
       << (node.body ? node.body->visit(*this) : "") << " }";
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

  std::vector<double> operator()(const ComparisonNode<double>& node) override {
    callCount += 1;
    double lhs = this->process(node.left)[0];
    double rhs = this->process(node.right)[0];
    bool result = false;
    switch (node.predicate) {
      case ComparisonPredicate::LT:
        result = lhs < rhs;
        break;
      case ComparisonPredicate::LE:
        result = lhs <= rhs;
        break;
      case ComparisonPredicate::GT:
        result = lhs > rhs;
        break;
      case ComparisonPredicate::GE:
        result = lhs >= rhs;
        break;
      case ComparisonPredicate::EQ:
        result = lhs == rhs;
        break;
      case ComparisonPredicate::NE:
        result = lhs != rhs;
        break;
    }
    return {result ? 1.0 : 0.0};
  }

  std::vector<double> operator()(const IfElseNode<double>& node) override {
    callCount += 1;
    double condition = this->process(node.condition)[0];
    if (condition != 0.0) {
      return this->process(node.thenBody);
    }
    return this->process(node.elseBody);
  }

  std::vector<double> operator()(const PowerNode<double>& node) override {
    callCount += 1;
    return {std::pow(this->process(node.base)[0], node.exponent)};
  }

  std::vector<double> operator()(const SplatNode& node) override {
    callCount += 1;
    return {node.value};
  }

  std::vector<double> operator()(const VariableNode<double>& node) override {
    callCount += 1;
    assert(node.value.has_value() &&
           "VariableNode must have a value during evaluation");
    return {node.value.value()};
  }

  std::vector<double> operator()(const YieldNode<double>& node) override {
    callCount += 1;
    std::vector<double> results;
    for (const auto& value : node.elements) {
      results.push_back(this->process(value)[0]);
    }
    return results;
  }

  std::vector<double> operator()(const ResultAtNode<double>& node) override {
    callCount += 1;
    std::vector<double> results = this->process(node.operand);
    return {results[node.index]};
  }

  std::vector<double> operator()(const ForLoopNode<double>& node) override {
    callCount += 1;
    std::vector<double> results;
    for (const auto& init : node.inits) {
      results.push_back(this->process(init)[0]);
    }
    for (size_t i = node.lower; i < node.upper; i += node.step) {
      // Set the induction variable value
      auto& inductionVarNode =
          std::get<VariableNode<double>>(node.inductionVar->node_variant);
      inductionVarNode.value = static_cast<double>(i);

      // Set the iter arg values
      for (size_t j = 0; j < node.iterArgs.size(); ++j) {
        auto& iterArgNode =
            std::get<VariableNode<double>>(node.iterArgs[j]->node_variant);
        iterArgNode.value = results[j];
      }

      // Clear the cache for the body and its dependencies before each iteration
      // since the body depends on variables that change each iteration
      if (node.body) {
        this->clearSubtreeCache(node.body);
        std::vector<double> bodyResults = this->process(node.body);
        for (size_t j = 0; j < results.size(); ++j) {
          results[j] = bodyResults[j];
        }
      }
    }
    return results;
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
  // Use a double print style for the left shift index because, while you could
  // type switch in FlattenedStringVisitor, there is no point in doing that for
  // this test.
  EXPECT_EQ(result, "(((x + 3.00) * (y ^ 2)) << 7.00)");
}

TEST(ArithmeticDagTest, TestDiv) {
  auto root = StringLeavedDag::leftRotate(
      StringLeavedDag::mul(
          StringLeavedDag::floorDiv(StringLeavedDag::leaf("x"), 3),
          StringLeavedDag::power(StringLeavedDag::leaf("y"), 2)),
      7);

  FlattenedStringVisitor visitor;
  std::string result = root->visit(visitor);
  EXPECT_EQ(result, "(((x / 3) * (y ^ 2)) << 7.00)");
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

TEST(ArithmeticDagTest, TestLoop) {
  auto x = DoubleLeavedDag::leaf(1.0);
  auto loop = DoubleLeavedDag::loop(x, {DagType::floatTy(64)}, 0, 10, 1);
  auto& loopNode = std::get<ForLoopNode<double>>(loop->node_variant);
  loopNode.body =
      DoubleLeavedDag::yield({DoubleLeavedDag::add(x, loopNode.iterArgs[0])});
  EvalVisitor visitor;
  double result = loop->visit(visitor)[0];
  EXPECT_EQ(result, 11);
}

TEST(ArithmeticDagTest, TestLoopStringVisitor) {
  auto x = StringLeavedDag::leaf("x");
  auto loop = StringLeavedDag::loop(
      x, {DagType::index()}, 0, 10, 1,
      [](const std::shared_ptr<StringLeavedDag>& inductionVar,
         const std::shared_ptr<StringLeavedDag>& iterArg) {
        return StringLeavedDag::yield({StringLeavedDag::add(
            StringLeavedDag::mul(inductionVar, StringLeavedDag::leaf("y")),
            iterArg)});
      });

  FlattenedStringVisitor visitor;
  std::string result = loop->visit(visitor);
  EXPECT_EQ(result, "for(i0=0 to 10 step 1; i1=x) { ((i0 * y) + i1) }");
}

TEST(ArithmeticDagTest, TestComparison) {
  auto x = DoubleLeavedDag::leaf(1.0);
  auto y = DoubleLeavedDag::leaf(2.0);
  auto lt = DoubleLeavedDag::comparison(x, y, ComparisonPredicate::LT);
  auto gt = DoubleLeavedDag::comparison(x, y, ComparisonPredicate::GT);

  EvalVisitor visitor;
  EXPECT_EQ(visitor.process(lt)[0], 1.0);
  EXPECT_EQ(visitor.process(gt)[0], 0.0);
}

TEST(ArithmeticDagTest, TestIfElse) {
  auto x = DoubleLeavedDag::leaf(1.0);
  auto y = DoubleLeavedDag::leaf(2.0);
  auto cond = DoubleLeavedDag::comparison(x, y, ComparisonPredicate::LT);
  auto ifNode = DoubleLeavedDag::ifElse(
      cond, [&]() { return DoubleLeavedDag::yield({x}); },
      [&]() { return DoubleLeavedDag::yield({y}); });

  EvalVisitor visitor;
  EXPECT_EQ(visitor.process(ifNode)[0], 1.0);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
