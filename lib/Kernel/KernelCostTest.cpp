#include <memory>
#include <unordered_set>

#include "gtest/gtest.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

// Visitor to count UNIQUE rotations in a symbolic DAG with CSE deduplication
class RotationCountVisitor {
 public:
  RotationCountVisitor() {}

  int64_t process(const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& node) {
    visitedNodes.clear();
    return processInternal(node);
  }

 private:
  std::unordered_set<const ArithmeticDagNode<SymbolicValue>*> visitedNodes;

  int64_t processInternal(const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& node) {
    const auto* nodePtr = node.get();

    // If we've already visited this node, don't count it again (CSE)
    if (visitedNodes.count(nodePtr)) {
      return 0;
    }
    visitedNodes.insert(nodePtr);

    return std::visit([this](auto&& arg) -> int64_t {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, ConstantScalarNode>) {
        return 0;
      } else if constexpr (std::is_same_v<T, ConstantTensorNode>) {
        return 0;
      } else if constexpr (std::is_same_v<T, LeafNode<SymbolicValue>>) {
        return 0;
      } else if constexpr (std::is_same_v<T, AddNode<SymbolicValue>>) {
        return processInternal(arg.left) + processInternal(arg.right);
      } else if constexpr (std::is_same_v<T, SubtractNode<SymbolicValue>>) {
        return processInternal(arg.left) + processInternal(arg.right);
      } else if constexpr (std::is_same_v<T, MultiplyNode<SymbolicValue>>) {
        return processInternal(arg.left) + processInternal(arg.right);
      } else if constexpr (std::is_same_v<T, LeftRotateNode<SymbolicValue>>) {
        return processInternal(arg.operand) + 1;
      } else if constexpr (std::is_same_v<T, ExtractNode<SymbolicValue>>) {
        return processInternal(arg.operand);
      }
      return 0;
    }, node->node_variant);
  }
};

TEST(KernelCostTest, MatvecDiagonal_4x4_CostIs4) {
  // 4x4 matrix-vector multiplication
  SymbolicValue matrix({4, 4});
  SymbolicValue vector({4});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  // Diagonal algorithm: one rotation per row
  EXPECT_EQ(rotations, 4);
}

TEST(KernelCostTest, MatvecDiagonal_100x100_CostIs100) {
  SymbolicValue matrix({100, 100});
  SymbolicValue vector({100});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  EXPECT_EQ(rotations, 100);
}

TEST(KernelCostTest, MatvecDiagonal_Rectangular_8x4) {
  // Non-square: 8 rows x 4 cols
  SymbolicValue matrix({8, 4});
  SymbolicValue vector({4});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  // Cost scales with number of rows (diagonals)
  EXPECT_EQ(rotations, 8);
}

TEST(KernelCostTest, MatvecDiagonal_Rectangular_4x8) {
  // 4 rows x 8 cols (wide matrix)
  SymbolicValue matrix({4, 8});
  SymbolicValue vector({8});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  // Cost scales with number of rows, not columns
  EXPECT_EQ(rotations, 4);
}

TEST(KernelCostTest, MatvecDiagonal_SmallMatrix_2x2) {
  SymbolicValue matrix({2, 2});
  SymbolicValue vector({2});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  EXPECT_EQ(rotations, 2);
}

TEST(KernelCostTest, MatvecDiagonal_LargeMatrix_512x512) {
  SymbolicValue matrix({512, 512});
  SymbolicValue vector({512});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  EXPECT_EQ(rotations, 512);
}

// Test that symbolic execution doesn't compute actual values
TEST(KernelCostTest, SymbolicValueHasShapeOnly) {
  SymbolicValue sym({10, 20});

  auto shape = sym.getShape();

  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], 10);
  EXPECT_EQ(shape[1], 20);
}

// Test DAG caching works (common subexpressions counted once)
TEST(KernelCostTest, CachingVisitorDeduplicatesSubexpressions) {
  SymbolicValue value({4});

  // Create DAG with shared subexpression:
  //      Add
  //     /   \
  //   Rot   Rot  (same rotation used twice)
  //    |     |
  //    v     v
  auto rotNode = ArithmeticDagNode<SymbolicValue>::leftRotate(
      ArithmeticDagNode<SymbolicValue>::leaf(value), 1);
  auto dag = ArithmeticDagNode<SymbolicValue>::add(rotNode, rotNode);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  // Should count the shared rotation only once due to caching
  EXPECT_EQ(rotations, 1);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
