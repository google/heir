#include "lib/Kernel/TestingUtils.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace kernel {

std::string PrintVisitor::operator()(const LeafNode<LiteralValue>& node) {
  const auto& nodeVal = node.value.get();
  const auto* vecVal = std::get_if<std::vector<int>>(&nodeVal);
  const auto* matVal = std::get_if<std::vector<std::vector<int>>>(&nodeVal);
  if (vecVal) {
    assert(vecVal->size() == node.value.getShape()[0]);
  }
  if (matVal) {
    assert(matVal->size() == node.value.getShape()[0]);
  }

  // just give a name to the vec
  if (vecVal) {
    return "v";
  }

  if (matVal) {
    return "Mat(...)";
  }

  return "UnknownLeaf";
}

std::string PrintVisitor::operator()(const ConstantScalarNode& node) {
  return std::to_string(node.value);
}

std::string PrintVisitor::operator()(const SplatNode& node) {
  return "splat(" + std::to_string(node.value) + ")";
}

std::string PrintVisitor::operator()(const AddNode<LiteralValue>& node) {
  std::string left = this->process(node.left);
  std::string right = this->process(node.right);
  return "(" + left + " + " + right + ")";
}

std::string PrintVisitor::operator()(const SubtractNode<LiteralValue>& node) {
  std::string left = this->process(node.left);
  std::string right = this->process(node.right);
  return "(" + left + " - " + right + ")";
}

std::string PrintVisitor::operator()(const MultiplyNode<LiteralValue>& node) {
  std::string left = this->process(node.left);
  std::string right = this->process(node.right);
  return "(" + left + " * " + right + ")";
}

std::string PrintVisitor::operator()(const FloorDivNode<LiteralValue>& node) {
  std::string left = this->process(node.left);
  return "(" + left + " / " + std::to_string(node.divisor) + ")";
}

std::string PrintVisitor::operator()(const LeftRotateNode<LiteralValue>& node) {
  std::string operand = this->process(node.operand);
  std::string shift = this->process(node.shift);
  return "Rot(" + operand + ", " + shift + ")";
}

std::string PrintVisitor::operator()(const ExtractNode<LiteralValue>& node) {
  // In these tests, extracting will always be from a plaintext matrix,
  // and the textual form of the entire matrix is too verbose. Could also
  // run a simplification on the generated kernel to inline the extracted
  // tensor instead of printing recursively.
  std::string indexStr = this->process(node.index);
  return "pt(" + indexStr + ")";
}

std::string PrintVisitor::operator()(const ComparisonNode<LiteralValue>& node) {
  std::string left = this->process(node.left);
  std::string right = this->process(node.right);
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
  return "(" + left + " " + op + " " + right + ")";
}

std::string PrintVisitor::operator()(const IfElseNode<LiteralValue>& node) {
  std::string condition = this->process(node.condition);
  std::string thenBody = this->process(node.thenBody);
  std::string elseBody = this->process(node.elseBody);
  return "if(" + condition + ") { " + thenBody + " } else { " + elseBody + " }";
}

std::string PrintVisitor::operator()(const VariableNode<LiteralValue>& node) {
  if (node.value.has_value()) {
    // If the variable has a value, print it
    const auto& val = node.value.value().get();
    if (const auto* intVal = std::get_if<int>(&val)) {
      return std::to_string(*intVal);
    }
    return "var(?)";
  }
  return "var";
}

std::string PrintVisitor::operator()(const ForLoopNode<LiteralValue>& node) {
  std::string result = "for(";
  result += std::to_string(node.lower) + ".." + std::to_string(node.upper);
  result += " step " + std::to_string(node.step) + ")";
  return result;
}

std::string PrintVisitor::operator()(const YieldNode<LiteralValue>& node) {
  if (node.elements.size() == 1) {
    return this->process(node.elements[0]);
  }
  std::string result = "yield(";
  for (size_t i = 0; i < node.elements.size(); ++i) {
    if (i > 0) result += ", ";
    result += this->process(node.elements[i]);
  }
  result += ")";
  return result;
}

std::string PrintVisitor::operator()(const ResultAtNode<LiteralValue>& node) {
  std::string operand = this->process(node.operand);
  return operand + "[" + std::to_string(node.index) + "]";
}

std::string printKernel(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag) {
  PrintVisitor visitor;
  return visitor.process(dag);
}

double evalMultiplicativeDepth(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag) {
  MultiplicativeDepthVisitorImpl visitor;
  return visitor.process(dag);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
