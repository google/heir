#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Utils/ArithmeticDag.h"
#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {
namespace {

// A visitor that evaluates an arithmetic DAG of ciphertext semantic tensors.
// The evaluation is done by replacing the leaves with their literal values and
// then computing the operations.
class EvalVisitor : public CachingVisitor<ValueOrLiteral, ValueOrLiteral> {
 public:
  using CachingVisitor<ValueOrLiteral, ValueOrLiteral>::operator();

  EvalVisitor() : CachingVisitor<ValueOrLiteral, ValueOrLiteral>() {}

  ValueOrLiteral operator()(const LeafNode<ValueOrLiteral>& node) {
    assert(!std::holds_alternative<Value>(node.value.value) &&
           "Encountered an mlir::Value during evaluation");
    const auto& nodeVal = std::get<CiphertextSemanticTensor>(node.value.value);
    const auto* vecVal = std::get_if<std::vector<double>>(&nodeVal);
    const auto* matVal =
        std::get_if<std::vector<std::vector<double>>>(&nodeVal);
    if (vecVal) {
      assert(vecVal->size() == node.value.shape[0]);
    }
    if (matVal) {
      assert(matVal->size() == node.value.shape[0]);
    }
    return {node.value.shape,
            std::get<CiphertextSemanticTensor>(node.value.value)};
  }

  ValueOrLiteral operator()(const AddNode<ValueOrLiteral>& node) {
    // Recursive calls use the public `process` method from the base class
    // to ensure caching is applied at every step.
    auto left = this->process(node.left);
    auto right = this->process(node.right);
    auto dim = left.shape[0];
    const auto& l_val = std::get<CiphertextSemanticTensor>(left.value);
    const auto& r_val = std::get<CiphertextSemanticTensor>(right.value);
    const auto* l_vec = std::get_if<std::vector<double>>(&l_val);
    const auto* r_vec = std::get_if<std::vector<double>>(&r_val);
    assert(l_vec && r_vec && "unsupported add operands");
    assert(left.shape == right.shape && "disagreeing shapes");
    std::vector<double> result(dim);
    for (size_t i = 0; i < dim; ++i) {
      result[i] = (*l_vec)[i] + (*r_vec)[i];
    }
    return {{dim}, result};
  }

  ValueOrLiteral operator()(const SubtractNode<ValueOrLiteral>& node) {
    auto left = this->process(node.left);
    auto right = this->process(node.right);
    auto dim = left.shape[0];
    const auto& l_val = std::get<CiphertextSemanticTensor>(left.value);
    const auto& r_val = std::get<CiphertextSemanticTensor>(right.value);
    const auto* l_vec = std::get_if<std::vector<double>>(&l_val);
    const auto* r_vec = std::get_if<std::vector<double>>(&r_val);
    assert(l_vec && r_vec && "unsupported sub operands");
    assert(left.shape == right.shape && "disagreeing shapes");
    std::vector<double> result(dim);
    for (size_t i = 0; i < dim; ++i) {
      result[i] = (*l_vec)[i] - (*r_vec)[i];
    }
    return {{dim}, result};
  }

  ValueOrLiteral operator()(const MultiplyNode<ValueOrLiteral>& node) {
    auto left = this->process(node.left);
    auto right = this->process(node.right);
    auto dim = left.shape[0];
    const auto& l_val = std::get<CiphertextSemanticTensor>(left.value);
    const auto& r_val = std::get<CiphertextSemanticTensor>(right.value);
    const auto* l_vec = std::get_if<std::vector<double>>(&l_val);
    const auto* r_vec = std::get_if<std::vector<double>>(&r_val);
    assert(l_vec && r_vec && "unsupported mul operands");
    assert(left.shape == right.shape && "disagreeing shapes");
    std::vector<double> result(dim);
    for (size_t i = 0; i < dim; ++i) {
      result[i] = (*l_vec)[i] * (*r_vec)[i];
    }
    return {{dim}, result};
  }

  // Cyclic left-rotation by a given index
  ValueOrLiteral operator()(const LeftRotateNode<ValueOrLiteral>& node) {
    auto operand = this->process(node.operand);
    auto dim = operand.shape[0];
    int amount = node.shift;
    const auto& o_val = std::get<CiphertextSemanticTensor>(operand.value);
    const auto* o_vec = std::get_if<std::vector<double>>(&o_val);
    assert(o_vec && "unsupported rotate operand");
    std::vector<double> result(dim);
    for (size_t i = 0; i < dim; ++i) {
      result[i] = (*o_vec)[(i + amount) % o_vec->size()];
    }
    return {{dim}, result};
  }

  ValueOrLiteral operator()(const ExtractNode<ValueOrLiteral>& node) {
    auto tensor = this->process(node.operand);
    unsigned index = node.index;
    return std::visit(
        [&](auto&& t) -> ValueOrLiteral {
          // We can only extract from a 2D vector.
          if constexpr (std::is_same_v<std::decay_t<decltype(t)>,
                                       std::vector<std::vector<double>>>) {
            return {{tensor.shape[0]}, t[index]};
          }
          assert(false && "Unsupported type for extraction");
          return {};
        },
        std::get<CiphertextSemanticTensor>(tensor.value));
  }
};

ValueOrLiteral evalKernel(
    const std::shared_ptr<ArithmeticDagNode<ValueOrLiteral>>& dag) {
  EvalVisitor visitor;
  return visitor.process(dag);
}

TEST(KernelImplementationTest, TestHaleviShoupMatvec) {
  std::vector<double> vector = {0, 1, 2, 3};
  // Pre-packed diagonally
  std::vector<std::vector<double>> matrix = {
      {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7, 8, 13}, {3, 4, 9, 14}};
  std::vector<double> expected = {14, 38, 62, 86};
  ValueOrLiteral matrixInput = {{4, 4}, matrix};
  ValueOrLiteral vectorInput = {{4}, vector};

  auto dag =
      implementKernel(KernelName::MatvecDiagonal, matrixInput, vectorInput);
  std::vector<double> actual = std::get<std::vector<double>>(
      std::get<CiphertextSemanticTensor>(evalKernel(dag).value));
  EXPECT_EQ(expected, actual);
}

TEST(KernelImplementationTest, TestExtract) {
  std::vector<std::vector<double>> matrix = {
      {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7, 8, 13}, {3, 4, 9, 14}};
  std::vector<double> expected = {1, 6, 11, 12};
  ValueOrLiteral matrixInput = {{4, 4}, matrix};

  auto dag = ArithmeticDagNode<ValueOrLiteral>::extract(
      ArithmeticDagNode<ValueOrLiteral>::leaf(matrixInput), 1);
  ValueOrLiteral actual = evalKernel(dag);
  EXPECT_EQ(std::get<std::vector<double>>(
                std::get<CiphertextSemanticTensor>(actual.value)),
            expected);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
