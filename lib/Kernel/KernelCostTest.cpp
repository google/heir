#include <memory>

#include "gtest/gtest.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

// Visitor to count rotations in a symbolic DAG
class RotationCountVisitor : public CachingVisitor<SymbolicValue, int64_t> {
 public:
  using CachingVisitor<SymbolicValue, int64_t>::operator();

  RotationCountVisitor() : CachingVisitor<SymbolicValue, int64_t>() {}

  int64_t operator()(const ConstantScalarNode& node) override { return 0; }

  int64_t operator()(const ConstantTensorNode& node) override { return 0; }

  int64_t operator()(const LeafNode<SymbolicValue>& node) override {
    return 0;
  }

  int64_t operator()(const AddNode<SymbolicValue>& node) override {
    return this->process(node.left) + this->process(node.right);
  }

  int64_t operator()(const SubtractNode<SymbolicValue>& node) override {
    return this->process(node.left) + this->process(node.right);
  }

  int64_t operator()(const MultiplyNode<SymbolicValue>& node) override {
    return this->process(node.left) + this->process(node.right);
  }

  int64_t operator()(const LeftRotateNode<SymbolicValue>& node) override {
    return this->process(node.operand) + 1;
  }

  int64_t operator()(const ExtractNode<SymbolicValue>& node) override {
    return this->process(node.operand);
  }
};

TEST(KernelCostTest, MatvecDiagonal_4x4_CostIs4) {
  // 4x4 matrix-vector multiplication
  SymbolicValue matrix({4, 4});
  SymbolicValue vector({4});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = dag->visit(counter);

  // Diagonal algorithm: one rotation per row
  EXPECT_EQ(rotations, 4);
}

TEST(KernelCostTest, MatvecDiagonal_100x100_CostIs100) {
  SymbolicValue matrix({100, 100});
  SymbolicValue vector({100});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = dag->visit(counter);

  EXPECT_EQ(rotations, 100);
}

TEST(KernelCostTest, MatvecDiagonal_Rectangular_8x4) {
  // Non-square: 8 rows x 4 cols
  SymbolicValue matrix({8, 4});
  SymbolicValue vector({4});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = dag->visit(counter);

  // Cost scales with number of rows (diagonals)
  EXPECT_EQ(rotations, 8);
}

TEST(KernelCostTest, MatvecDiagonal_Rectangular_4x8) {
  // 4 rows x 8 cols (wide matrix)
  SymbolicValue matrix({4, 8});
  SymbolicValue vector({8});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = dag->visit(counter);

  // Cost scales with number of rows, not columns
  EXPECT_EQ(rotations, 4);
}

TEST(KernelCostTest, MatvecDiagonal_SmallMatrix_2x2) {
  SymbolicValue matrix({2, 2});
  SymbolicValue vector({2});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = dag->visit(counter);

  EXPECT_EQ(rotations, 2);
}

TEST(KernelCostTest, MatvecDiagonal_LargeMatrix_512x512) {
  SymbolicValue matrix({512, 512});
  SymbolicValue vector({512});

  auto dag = implementMatvec(KernelName::MatvecDiagonal, matrix, vector);

  RotationCountVisitor counter;
  int64_t rotations = dag->visit(counter);

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
  int64_t rotations = dag->visit(counter);

  // Should count the shared rotation only once due to caching
  EXPECT_EQ(rotations, 1);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
