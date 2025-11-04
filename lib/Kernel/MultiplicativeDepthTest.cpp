#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/MultiplicativeDepthVisitor.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

TEST(MultiplicativeDepthTest, SimpleAddition) {
  // Addition doesn't increase depth
  SymbolicValue a({4}, true);
  SymbolicValue b({4}, true);

  auto leftLeaf = ArithmeticDagNode<SymbolicValue>::leaf(a);
  auto rightLeaf = ArithmeticDagNode<SymbolicValue>::leaf(b);
  auto addNode = ArithmeticDagNode<SymbolicValue>::add(leftLeaf, rightLeaf);

  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(addNode);

  // Addition has depth 0
  EXPECT_EQ(depth, 0);
}

TEST(MultiplicativeDepthTest, SimpleMultiplication) {
  // Multiplication increases depth by 1
  SymbolicValue a({4}, true);
  SymbolicValue b({4}, true);

  auto leftLeaf = ArithmeticDagNode<SymbolicValue>::leaf(a);
  auto rightLeaf = ArithmeticDagNode<SymbolicValue>::leaf(b);
  auto mulNode = ArithmeticDagNode<SymbolicValue>::mul(leftLeaf, rightLeaf);

  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(mulNode);

  // One multiplication has depth 1
  EXPECT_EQ(depth, 1);
}

TEST(MultiplicativeDepthTest, TwoSequentialMultiplications) {
  // (a * b) * c should have depth 2
  SymbolicValue a({4}, true);
  SymbolicValue b({4}, true);
  SymbolicValue c({4}, true);

  auto aLeaf = ArithmeticDagNode<SymbolicValue>::leaf(a);
  auto bLeaf = ArithmeticDagNode<SymbolicValue>::leaf(b);
  auto cLeaf = ArithmeticDagNode<SymbolicValue>::leaf(c);

  auto mul1 = ArithmeticDagNode<SymbolicValue>::mul(aLeaf, bLeaf);
  auto mul2 = ArithmeticDagNode<SymbolicValue>::mul(mul1, cLeaf);

  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(mul2);

  // Two sequential multiplications: depth 2
  EXPECT_EQ(depth, 2);
}

TEST(MultiplicativeDepthTest, ParallelMultiplications) {
  // (a * b) + (c * d) should have depth 1
  // Addition doesn't increase depth, just takes max
  SymbolicValue a({4}, true);
  SymbolicValue b({4}, true);
  SymbolicValue c({4}, true);
  SymbolicValue d({4}, true);

  auto aLeaf = ArithmeticDagNode<SymbolicValue>::leaf(a);
  auto bLeaf = ArithmeticDagNode<SymbolicValue>::leaf(b);
  auto cLeaf = ArithmeticDagNode<SymbolicValue>::leaf(c);
  auto dLeaf = ArithmeticDagNode<SymbolicValue>::leaf(d);

  auto mul1 = ArithmeticDagNode<SymbolicValue>::mul(aLeaf, bLeaf);
  auto mul2 = ArithmeticDagNode<SymbolicValue>::mul(cLeaf, dLeaf);
  auto addNode = ArithmeticDagNode<SymbolicValue>::add(mul1, mul2);

  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(addNode);

  // Parallel multiplications: depth is max(1, 1) = 1
  EXPECT_EQ(depth, 1);
}

TEST(MultiplicativeDepthTest, RotationDoesNotIncreaseDepth) {
  // Rotation is an automorphism, doesn't increase depth
  SymbolicValue a({4}, true);
  SymbolicValue b({4}, true);

  auto aLeaf = ArithmeticDagNode<SymbolicValue>::leaf(a);
  auto bLeaf = ArithmeticDagNode<SymbolicValue>::leaf(b);
  auto mulNode = ArithmeticDagNode<SymbolicValue>::mul(aLeaf, bLeaf);
  auto rotNode = ArithmeticDagNode<SymbolicValue>::leftRotate(mulNode, 1);

  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(rotNode);

  // Rotation doesn't add depth: still 1 from multiplication
  EXPECT_EQ(depth, 1);
}

TEST(MultiplicativeDepthTest, PowerOperation) {
  // x^8 requires ceil(log2(8)) = 3 multiplications using repeated squaring
  SymbolicValue x({4}, true);

  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);
  auto powerNode = ArithmeticDagNode<SymbolicValue>::power(xLeaf, 8);

  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(powerNode);

  // x^8 = ((x^2)^2)^2 requires 3 multiplications
  EXPECT_EQ(depth, 3);
}

TEST(MultiplicativeDepthTest, ComplexCircuit) {
  // Test a more complex circuit: ((a * b) + c) * d
  // Depth should be 2: one mul for (a*b), then another for result * d
  SymbolicValue a({4}, true);
  SymbolicValue b({4}, true);
  SymbolicValue c({4}, true);
  SymbolicValue d({4}, true);

  auto aLeaf = ArithmeticDagNode<SymbolicValue>::leaf(a);
  auto bLeaf = ArithmeticDagNode<SymbolicValue>::leaf(b);
  auto cLeaf = ArithmeticDagNode<SymbolicValue>::leaf(c);
  auto dLeaf = ArithmeticDagNode<SymbolicValue>::leaf(d);

  auto mul1 = ArithmeticDagNode<SymbolicValue>::mul(aLeaf, bLeaf);
  auto add1 = ArithmeticDagNode<SymbolicValue>::add(mul1, cLeaf);
  auto mul2 = ArithmeticDagNode<SymbolicValue>::mul(add1, dLeaf);

  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(mul2);

  // Depth is 2: one from mul1, one more from mul2
  EXPECT_EQ(depth, 2);
}

TEST(MultiplicativeDepthTest, HaleviShoupMatvecDepth) {
  // Test depth of Halevi-Shoup kernel (should be 1 - only multiplications with plaintext)
  SymbolicValue vector({4}, true);   // Vector is encrypted (secret)
  SymbolicValue matrix({4, 4}, false);  // Matrix is plaintext
  std::vector<int64_t> originalShape = {4, 4};

  auto dag = implementHaleviShoup(vector, matrix, originalShape);

  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(dag);

  // Halevi-Shoup uses only ciphertext-plaintext multiplications
  // Depth should be 1
  EXPECT_EQ(depth, 1);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
