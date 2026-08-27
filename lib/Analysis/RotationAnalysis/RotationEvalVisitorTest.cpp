#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Analysis/RotationAnalysis/RotationEvalVisitor.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"

namespace mlir {
namespace heir {

using kernel::ArithmeticDagNode;
using kernel::DagType;
using kernel::LiteralValue;
using ::testing::UnorderedElementsAre;

namespace {

using Node = ArithmeticDagNode<LiteralValue>;
using NodePtr = std::shared_ptr<Node>;

TEST(RotationEvalVisitorTest, TestSimpleLoop) {
  LiteralValue inputVector({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto x = Node::leaf(inputVector);
  auto two = Node::constantScalar(2, DagType::index());
  auto loop = Node::loop(x, {DagType::intTensor(32, {10})}, 0, 5, 1,
                         [&](NodePtr iv, NodePtr iterArg) {
                           NodePtr mulBy2 = Node::mul(two, iv);
                           NodePtr iterRot = Node::leftRotate(iterArg, mulBy2);
                           return Node::yield({iterRot});
                         });
  EXPECT_THAT(evalRotations(loop), UnorderedElementsAre(0, 2, 4, 6, 8));
}

// A loop that rotates by both the induction variable and by
// inductionVar + constant. This is the shape a rolled convolution emits, and
// the offset rotations are the ones whose Galois keys went missing (a rotation
// by 65 = 1 + 64 aborted lattigo with "GaloisKey[39685] is nil").
TEST(RotationEvalVisitorTest, TestLoopWithOffsetRotation) {
  LiteralValue inputVector({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto x = Node::leaf(inputVector);
  auto four = Node::constantScalar(4, DagType::index());
  auto loop = Node::loop(x, {DagType::intTensor(32, {10})}, 1, 6, 1,
                         [&](NodePtr iv, NodePtr iterArg) {
                           NodePtr plain = Node::leftRotate(iterArg, iv);
                           NodePtr offset = Node::add(iv, four);
                           NodePtr shifted = Node::leftRotate(plain, offset);
                           return Node::yield({shifted});
                         });
  // iv runs 1..5, so the plain rotations are 1..5 and the offset rotations are
  // 5..9; normalized modulo the 10-slot dimension they stay distinct.
  EXPECT_THAT(evalRotations(loop),
              UnorderedElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9));
}

class RollUnrollTest : public testing::TestWithParam<bool> {};

TEST_P(RollUnrollTest, RotateAndReduceKernel) {
  int64_t n = 10;
  int64_t period = 1;

  std::vector<int> vector = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<std::vector<int>> plaintexts = {
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
      {3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
      {4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
      {5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
      {6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
      {9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
      {10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
  };

  LiteralValue vectorInput(vector);
  std::shared_ptr<ArithmeticDagNode<LiteralValue>> result;
  std::optional<LiteralValue> plaintextsInput =
      std::optional<LiteralValue>(LiteralValue(plaintexts));
  auto dag = implementRotateAndReduce(vectorInput, plaintextsInput, period, n,
                                      DagType::intTensor(32, {n}), {},
                                      "arith.addi", /* unroll= */ GetParam());
  EXPECT_THAT(evalRotations(dag), UnorderedElementsAre(0, 1, 2, 3, 4, 6, 8));
}

INSTANTIATE_TEST_SUITE_P(WithAndWithoutRolledSuite, RollUnrollTest,
                         testing::Values(false, true));

}  // namespace
}  // namespace heir
}  // namespace mlir
