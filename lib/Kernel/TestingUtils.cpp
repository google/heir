#include "lib/Kernel/TestingUtils.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

EvalResults EvalVisitor::operator()(const LeafNode<LiteralValue>& node) {
  const auto& nodeVal = node.value.get();
  const auto* vecVal = std::get_if<std::vector<int>>(&nodeVal);
  const auto* matVal = std::get_if<std::vector<std::vector<int>>>(&nodeVal);
  if (vecVal) {
    assert(vecVal->size() == node.value.getShape()[0]);
  }
  if (matVal) {
    assert(matVal->size() == node.value.getShape()[0]);
  }

  return {node.value.get()};
}

EvalResults EvalVisitor::operator()(const AddNode<LiteralValue>& node) {
  // Recursive calls use the public `process` method from the base class
  // to ensure caching is applied at every step.
  auto left = this->process(node.left)[0];
  auto right = this->process(node.right)[0];
  const auto& lVal = left.get();
  const auto& rVal = right.get();

  // Handle scalar addition
  const auto* lScalar = std::get_if<int>(&lVal);
  const auto* rScalar = std::get_if<int>(&rVal);
  if (lScalar && rScalar) {
    return {LiteralValue(*lScalar + *rScalar)};
  }

  // Handle vector addition
  auto dim = left.getShape()[0];
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  const auto* rVec = std::get_if<std::vector<int>>(&rVal);
  assert(lVec && rVec && "unsupported add operands");
  assert(left.getShape() == right.getShape() && "disagreeing shapes");
  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*lVec)[i] + (*rVec)[i];
  }
  return {result};
}

EvalResults EvalVisitor::operator()(const SubtractNode<LiteralValue>& node) {
  auto left = this->process(node.left)[0];
  auto right = this->process(node.right)[0];
  const auto& lVal = left.get();
  const auto& rVal = right.get();

  // Handle scalar subtraction
  const auto* lScalar = std::get_if<int>(&lVal);
  const auto* rScalar = std::get_if<int>(&rVal);
  if (lScalar && rScalar) {
    return {LiteralValue(*lScalar - *rScalar)};
  }

  // Handle vector subtraction
  auto dim = left.getShape()[0];
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  const auto* rVec = std::get_if<std::vector<int>>(&rVal);
  assert(lVec && rVec && "unsupported sub operands");
  assert(left.getShape() == right.getShape() && "disagreeing shapes");
  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*lVec)[i] - (*rVec)[i];
  }

  return {result};
}

EvalResults EvalVisitor::operator()(const MultiplyNode<LiteralValue>& node) {
  auto left = this->process(node.left)[0];
  auto right = this->process(node.right)[0];
  const auto& lVal = left.get();
  const auto& rVal = right.get();

  // Handle scalar multiplication
  const auto* lScalar = std::get_if<int>(&lVal);
  const auto* rScalar = std::get_if<int>(&rVal);
  if (lScalar && rScalar) {
    return {LiteralValue(*lScalar * *rScalar)};
  }

  // Handle vector multiplication
  auto dim = left.getShape()[0];
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  const auto* rVec = std::get_if<std::vector<int>>(&rVal);
  assert(lVec && rVec && "unsupported mul operands");
  assert(left.getShape() == right.getShape() && "disagreeing shapes");
  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*lVec)[i] * (*rVec)[i];
  }
  return {result};
}

EvalResults EvalVisitor::operator()(const FloorDivNode<LiteralValue>& node) {
  auto left = this->process(node.left)[0];
  const auto& lVal = left.get();

  // Scalar case
  const auto* lScalar = std::get_if<int>(&lVal);
  if (*lScalar) {
    return {LiteralValue(*lScalar / node.divisor)};
  }

  // Vector case
  auto dim = left.getShape()[0];
  const auto* lVec = std::get_if<std::vector<int>>(&lVal);
  assert(lVec && "unsupported floorDiv operands");
  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*lVec)[i] / node.divisor;
  }
  return {result};
}

// Cyclic left-rotation by a given index
EvalResults EvalVisitor::operator()(const LeftRotateNode<LiteralValue>& node) {
  auto operand = this->process(node.operand)[0];
  auto dim = operand.getShape()[0];
  auto amount = node.shift;
  // Normalize amount to be in [0, dim)
  amount = ((amount % dim) + dim) % dim;

  const auto& oVal = operand.get();
  const auto* oVec = std::get_if<std::vector<int>>(&oVal);
  assert(oVec && "unsupported rotate operand");
  std::vector<int> result(dim);
  for (size_t i = 0; i < dim; ++i) {
    result[i] = (*oVec)[(i + amount) % oVec->size()];
  }
  return {result};
}

EvalResults EvalVisitor::operator()(const ExtractNode<LiteralValue>& node) {
  auto tensor = this->process(node.operand)[0];
  int index = node.index;

  return std::visit(
      [&](auto&& t) -> EvalResults {
        if constexpr (std::is_same_v<std::decay_t<decltype(t)>,
                                     std::vector<std::vector<int>>>) {
          return {LiteralValue(t[index])};
        } else if constexpr (std::is_same_v<std::decay_t<decltype(t)>,
                                            std::vector<int>>) {
          return {LiteralValue(t[index])};
        }
        assert(false && "Unsupported type for extraction");
        return {};
      },
      tensor.get());
}

EvalResults EvalVisitor::operator()(const ConstantTensorNode& node) {
  // A bit of a hack, only support ints in testing
  std::vector<int> vec;
  vec.reserve(node.value.size());
  for (double v : node.value) {
    vec.push_back(static_cast<int>(v));
  }
  return {LiteralValue(vec)};
}

EvalResults EvalVisitor::operator()(const ConstantScalarNode& node) {
  // A bit of a hack, casting the double to an int
  return {LiteralValue(static_cast<int>(node.value))};
}

EvalResults EvalVisitor::operator()(const SplatNode& node) {
  // A bit of a hack, casting the double to an int
  int splatValue = static_cast<int>(node.value);

  // Check if this is a tensor type
  if (std::holds_alternative<kernel::IntTensorType>(node.type.type_variant)) {
    const auto& tensorType =
        std::get<kernel::IntTensorType>(node.type.type_variant);
    // Compute total size as product of all dimensions
    int64_t totalSize = 1;
    for (int64_t dim : tensorType.shape) {
      totalSize *= dim;
    }
    std::vector<int> result(totalSize, splatValue);
    return {LiteralValue(result)};
  } else if (std::holds_alternative<kernel::FloatTensorType>(
                 node.type.type_variant)) {
    const auto& tensorType =
        std::get<kernel::FloatTensorType>(node.type.type_variant);
    // Compute total size as product of all dimensions
    int64_t totalSize = 1;
    for (int64_t dim : tensorType.shape) {
      totalSize *= dim;
    }
    std::vector<int> result(totalSize, splatValue);
    return {LiteralValue(result)};
  }

  // Scalar type
  return {LiteralValue(splatValue)};
}

EvalResults evalKernel(
    const std::shared_ptr<ArithmeticDagNode<LiteralValue>>& dag) {
  EvalVisitor visitor;
  return visitor.process(dag);
}

std::vector<EvalResults> multiEvalKernel(
    ArrayRef<std::shared_ptr<ArithmeticDagNode<LiteralValue>>> dags) {
  EvalVisitor visitor;
  return visitor.process(dags);
}

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
  std::string shift = std::to_string(node.shift);
  return "Rot(" + operand + ", " + shift + ")";
}

std::string PrintVisitor::operator()(const ExtractNode<LiteralValue>& node) {
  // In these tests, extracting will always be from a plaintext matrix,
  // and the textual form of the entire matrix is too verbose. Could also
  // run a simplification on the generated kernel to inline the extracted
  // tensor instead of printing recursively.
  std::string indexStr = std::to_string(node.index);
  return "pt(" + indexStr + ")";
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
