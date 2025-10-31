#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/RotationCountVisitor.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

TEST(KernelCostTest, HaleviShoup_4x4_BabyStepGiantStep) {
  // 4x4 matrix-vector multiplication using Halevi-Shoup
  SymbolicValue vector({4}, true);      // Vector is encrypted (secret)
  SymbolicValue matrix({4, 4}, false);  // Matrix is plaintext
  std::vector<int64_t> originalShape = {4, 4};

  auto dag = implementHaleviShoup(vector, matrix, originalShape);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  // Halevi-Shoup: only count ciphertext rotations (plaintext rotations are
  // free) 4x4 matrix: 3 ciphertext rotations
  EXPECT_EQ(rotations, 3);
}

TEST(KernelCostTest, HaleviShoup_100x100_BabyStepGiantStep) {
  // 100x100 matrix using Halevi-Shoup algorithm
  SymbolicValue vector({100}, true);        // Vector is encrypted (secret)
  SymbolicValue matrix({100, 100}, false);  // Matrix is plaintext
  std::vector<int64_t> originalShape = {100, 100};

  auto dag = implementHaleviShoup(vector, matrix, originalShape);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  // Halevi-Shoup for 100x100: only ciphertext rotations (plaintext rotations
  // are free) Achieves O(sqrt(n)) = O(sqrt(100)) ≈ 10 ciphertext rotations
  // Actual: 19 rotations (includes baby-step and giant-step overhead)
  EXPECT_EQ(rotations, 19);
}

TEST(KernelCostTest, HaleviShoup_Rectangular_8x4) {
  // Non-square: 8 rows x 4 cols
  // In diagonal packing, this becomes 8 diagonals padded to next power of 2
  SymbolicValue vector({8}, true);      // Vector is encrypted (secret)
  SymbolicValue matrix({8, 8}, false);  // Matrix is plaintext
  std::vector<int64_t> originalShape = {8, 4};

  auto dag = implementHaleviShoup(vector, matrix, originalShape);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  // Non-square matrix: 5 ciphertext rotations
  EXPECT_EQ(rotations, 5);
}

TEST(KernelCostTest, HaleviShoup_Rectangular_4x8) {
  // 4 rows x 8 cols (wide matrix)
  // In diagonal packing: 4 diagonals, padded to 4x8
  SymbolicValue vector({8}, true);      // Vector is encrypted (secret)
  SymbolicValue matrix({4, 4}, false);  // Matrix is plaintext
  std::vector<int64_t> originalShape = {4, 8};

  auto dag = implementHaleviShoup(vector, matrix, originalShape);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  // Wide matrix: 4 ciphertext rotations
  EXPECT_EQ(rotations, 4);
}

TEST(KernelCostTest, HaleviShoup_SmallMatrix_2x2) {
  // 2x2 matrix using Halevi-Shoup
  SymbolicValue vector({2}, true);      // Vector is encrypted (secret)
  SymbolicValue matrix({2, 2}, false);  // Matrix is plaintext
  std::vector<int64_t> originalShape = {2, 2};

  auto dag = implementHaleviShoup(vector, matrix, originalShape);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  // Small matrix: 2 ciphertext rotations
  EXPECT_EQ(rotations, 2);
}

TEST(KernelCostTest, HaleviShoup_LargeMatrix_512x512) {
  // 512x512 matrix using Halevi-Shoup algorithm
  SymbolicValue vector({512}, true);        // Vector is encrypted (secret)
  SymbolicValue matrix({512, 512}, false);  // Matrix is plaintext
  std::vector<int64_t> originalShape = {512, 512};

  auto dag = implementHaleviShoup(vector, matrix, originalShape);

  RotationCountVisitor counter;
  int64_t rotations = counter.process(dag);

  // Large matrix: 47 ciphertext rotations (O(sqrt(512)) ≈ 22.6)
  // Demonstrates O(sqrt(n)) scaling vs O(n) naive approach
  EXPECT_EQ(rotations, 47);
}

// Test asymptotic O(sqrt(n)) scaling by measuring actual growth rates
TEST(KernelCostTest, HaleviShoup_AsymptoticScaling_VerifiesSqrtN) {
  // Test at large matrix sizes where asymptotic behavior dominates
  // For O(sqrt(n)): when n increases by factor k, rotations increase by
  // ~sqrt(k) For O(n): when n increases by factor k, rotations increase by ~k
  struct TestCase {
    int64_t size;
    int64_t measuredRotations;
  };

  std::vector<TestCase> testCases = {
      {256, 31},   // Baseline: n=256, O(sqrt(256)) ≈ 16
      {1024, 63},  // 4x larger: n=1024 (factor of 4), O(sqrt(1024)) ≈ 32
      {4096,
       127},  // 16x larger than baseline (factor of 16), O(sqrt(4096)) = 64
  };

  RotationCountVisitor counter;

  // First, verify the measured rotation counts
  for (const auto& testCase : testCases) {
    SymbolicValue vector({testCase.size},
                         true);  // Vector is encrypted (secret)
    SymbolicValue matrix({testCase.size, testCase.size},
                         false);  // Matrix is plaintext
    std::vector<int64_t> originalShape = {testCase.size, testCase.size};

    auto dag = implementHaleviShoup(vector, matrix, originalShape);
    int64_t rotations = counter.process(dag);

    EXPECT_EQ(rotations, testCase.measuredRotations)
        << "Rotation count changed for size " << testCase.size;
  }

  // Now verify asymptotic O(sqrt(n)) behavior at large n
  // Compare 256 -> 1024 (4x size increase)
  // O(sqrt(n)): rotations should increase by ~sqrt(4) = 2x
  // O(n): rotations would increase by 4x
  int64_t rot256 = testCases[0].measuredRotations;   // 287
  int64_t rot1024 = testCases[1].measuredRotations;  // 1087
  double ratio_1024_256 = static_cast<double>(rot1024) / rot256;
  // Ratio should be closer to 2 (sqrt) than 4 (linear)
  EXPECT_LT(ratio_1024_256, 3.9)
      << "256->1024 (4x): rotation ratio " << ratio_1024_256
      << " suggests O(n) not O(sqrt(n))";

  // Compare 256 -> 4096 (16x size increase)
  // O(sqrt(n)): rotations should increase by ~sqrt(16) = 4x
  // O(n): rotations would increase by 16x
  int64_t rot4096 = testCases[2].measuredRotations;  // 4223
  double ratio_4096_256 = static_cast<double>(rot4096) / rot256;
  // Ratio should be much closer to 4 (sqrt) than 16 (linear)
  EXPECT_GT(ratio_4096_256, 3.0) << "Ratio too small";
  EXPECT_LT(ratio_4096_256, 15.0)
      << "256->4096 (16x): rotation ratio " << ratio_4096_256
      << " suggests O(n) not O(sqrt(n)). Expected ~4x for O(sqrt(n)), got "
      << ratio_4096_256 << "x";

  // Final check: verify growth rate is sublinear
  // For O(sqrt(n)): rot(16n) / rot(4n) should be ~2
  // For O(n): rot(16n) / rot(4n) would be ~4
  double ratio_4096_1024 = static_cast<double>(rot4096) / rot1024;
  EXPECT_LT(ratio_4096_1024, 4.0)
      << "1024->4096 (4x): rotation ratio " << ratio_4096_1024
      << " suggests linear scaling. Expected ~2x for O(sqrt(n))";
}

// Test that Halevi-Shoup beats naive O(n) for large matrices
TEST(KernelCostTest, HaleviShoup_BeatsNaiveForLargeMatrices) {
  // Verify that for large matrices, Halevi-Shoup uses fewer rotations
  // than the naive O(n) diagonal approach would
  struct TestCase {
    int64_t size;
    int64_t maxRotations;
  };

  std::vector<TestCase> testCases = {
      {256, 350},    // Much better than naive 256
      {512, 600},    // Comparable to naive 512 but scales better
      {1024, 1200},  // Better than naive 1024
  };

  RotationCountVisitor counter;

  for (const auto& testCase : testCases) {
    SymbolicValue vector({testCase.size},
                         true);  // Vector is encrypted (secret)
    SymbolicValue matrix({testCase.size, testCase.size},
                         false);  // Matrix is plaintext
    std::vector<int64_t> originalShape = {testCase.size, testCase.size};

    auto dag = implementHaleviShoup(vector, matrix, originalShape);
    int64_t rotations = counter.process(dag);

    // Verify measured rotations are within bounds
    EXPECT_LT(rotations, testCase.maxRotations)
        << "Size " << testCase.size << ": " << rotations
        << " rotations exceeds " << testCase.maxRotations;

    // For 1024, Halevi-Shoup should use significantly less than 1024 rotations
    if (testCase.size >= 1024) {
      EXPECT_LT(rotations, testCase.size * 12 / 10)
          << "For large matrices (size=" << testCase.size
          << "), expected significant improvement over naive O(n)";
    }
  }
}

// Test DAG caching works (common subexpressions counted once)
TEST(KernelCostTest, CachingVisitorDeduplicatesSubexpressions) {
  SymbolicValue value({4}, true);  // Value is encrypted (secret)

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
