#include <cstdint>
#include <iostream>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"
#include "lib/Dialect/TensorExt/Transforms/RotationGroupKernel.h"
#include "lib/Dialect/TensorExt/Transforms/ShiftScheme.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/TestingUtils.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {
namespace {

using kernel::LiteralValue;

std::vector<std::vector<int>> manuallyApplyMapping(
    const Mapping& mapping, const std::vector<std::vector<int>>& input,
    int64_t ctSize) {
  std::vector<std::vector<int>> output(input.size(),
                                       std::vector<int>(ctSize, 0));
  for (const auto& entry : mapping) {
    output[entry.target.ct][entry.target.slot] =
        input[entry.source.ct][entry.source.slot];
  }
  return output;
}

::testing::AssertionResult simulateShiftNetwork(const Mapping& mapping,
                                                const ShiftScheme& scheme,
                                                int64_t numCiphertexts,
                                                int64_t ciphertextSize) {
  // print the rotation groups
  std::cout << "Rotation groups:\n";
  for (const auto& row : scheme.rotationGroups) {
    for (const auto& slot : row) {
      std::cout << "(" << slot.ct << "," << slot.slot << ") ";
    }
    std::cout << "\n";
  }

  SmallVector<LiteralValue> inputLeaves;
  std::vector<std::vector<int>> input;
  input.reserve(numCiphertexts);
  inputLeaves.reserve(numCiphertexts);
  // row-major values as input
  for (int64_t i = 0; i < numCiphertexts; i++) {
    std::vector<int> oneInput(ciphertextSize);
    for (int64_t j = 0; j < ciphertextSize; j++) {
      oneInput[j] = i * ciphertextSize + j;
    }
    input.push_back(oneInput);
    inputLeaves.push_back(LiteralValue(oneInput));
  }

  // print the input
  std::cout << "Input:\n";
  for (const auto& row : input) {
    for (const auto& val : row) {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }

  auto expected = manuallyApplyMapping(mapping, input, ciphertextSize);
  auto dag =
      implementShiftNetwork(inputLeaves, mapping, scheme, ciphertextSize);
  std::vector<LiteralValue> actual = multiEvalKernel(dag);

  std::vector<std::vector<int>> combinedActual;
  combinedActual.reserve(numCiphertexts);
  for (const LiteralValue& val : actual) {
    combinedActual.push_back(std::get<std::vector<int>>(val.getTensor()));
  }

  ::testing::StringMatchResultListener listener;
  bool matches = ::testing::ExplainMatchResult(testing::ContainerEq(expected),
                                               combinedActual, &listener);
  if (matches) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << listener.str();
  }
}

::testing::AssertionResult checkMapping(const Mapping& mapping,
                                        int64_t numCiphertexts,
                                        int64_t ciphertextSize,
                                        unsigned naiveNumRGExpected = 0) {
  VosVosErkinShiftNetworks shiftNetworks;

  auto naiveScheme = shiftNetworks.findShiftScheme(mapping);
  unsigned naiveNumRG = naiveScheme.rotationGroups.size();
  if (naiveNumRGExpected > 0 && naiveNumRG != naiveNumRGExpected)
    return ::testing::AssertionFailure()
           << "Expected " << naiveNumRGExpected << " rotation groups but got "
           << naiveNumRG;
  auto naiveResult = simulateShiftNetwork(mapping, naiveScheme, numCiphertexts,
                                          ciphertextSize);
  if (!naiveResult) return naiveResult;

  // We try a large number of shift orders here such that we can be effectively
  // certain that we will find a network that is at least as good as the "naive"
  // one.
  auto bestScheme = shiftNetworks.findBestShiftScheme(mapping, 1000);
  unsigned bestNumRG = bestScheme.rotationGroups.size();
  if (bestNumRG > naiveNumRG)
    return ::testing::AssertionFailure()
           << "Expected best found network with " << bestNumRG
           << " rotation groups to not be worse then naive network which has "
           << naiveNumRG;
  auto bestResult =
      simulateShiftNetwork(mapping, bestScheme, numCiphertexts, ciphertextSize);
  if (!bestResult) return bestResult;

  return ::testing::AssertionSuccess();
}

TEST(ImplementShiftNetworkTest, TestTrivial) {
  int64_t numCts = 1;
  int64_t ctSize = 8;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(0, 0));
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

TEST(ImplementShiftNetworkTest, TestFig3) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(0, 13));
  mapping.add(CtSlot(0, 1), CtSlot(0, 8));
  mapping.add(CtSlot(0, 2), CtSlot(0, 4));
  mapping.add(CtSlot(0, 3), CtSlot(0, 0));
  mapping.add(CtSlot(0, 4), CtSlot(0, 11));
  mapping.add(CtSlot(0, 5), CtSlot(0, 7));
  mapping.add(CtSlot(0, 6), CtSlot(0, 14));
  mapping.add(CtSlot(0, 7), CtSlot(0, 5));
  mapping.add(CtSlot(0, 8), CtSlot(0, 15));
  mapping.add(CtSlot(0, 9), CtSlot(0, 3));
  mapping.add(CtSlot(0, 10), CtSlot(0, 12));
  mapping.add(CtSlot(0, 11), CtSlot(0, 6));
  mapping.add(CtSlot(0, 12), CtSlot(0, 10));
  mapping.add(CtSlot(0, 13), CtSlot(0, 2));
  mapping.add(CtSlot(0, 14), CtSlot(0, 9));
  mapping.add(CtSlot(0, 15), CtSlot(0, 1));
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

TEST(ImplementShiftNetworkTest, TestFullReplication) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(0, 0));
  mapping.add(CtSlot(0, 0), CtSlot(0, 1));
  mapping.add(CtSlot(0, 0), CtSlot(0, 2));
  mapping.add(CtSlot(0, 0), CtSlot(0, 3));
  mapping.add(CtSlot(0, 0), CtSlot(0, 4));
  mapping.add(CtSlot(0, 0), CtSlot(0, 5));
  mapping.add(CtSlot(0, 0), CtSlot(0, 6));
  mapping.add(CtSlot(0, 0), CtSlot(0, 7));
  mapping.add(CtSlot(0, 0), CtSlot(0, 8));
  mapping.add(CtSlot(0, 0), CtSlot(0, 9));
  mapping.add(CtSlot(0, 0), CtSlot(0, 10));
  mapping.add(CtSlot(0, 0), CtSlot(0, 11));
  mapping.add(CtSlot(0, 0), CtSlot(0, 12));
  mapping.add(CtSlot(0, 0), CtSlot(0, 13));
  mapping.add(CtSlot(0, 0), CtSlot(0, 14));
  mapping.add(CtSlot(0, 0), CtSlot(0, 15));
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

TEST(ImplementShiftNetworkTest, TestTwoReplication) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 14), CtSlot(0, 0));
  mapping.add(CtSlot(0, 14), CtSlot(0, 1));
  mapping.add(CtSlot(0, 14), CtSlot(0, 2));
  mapping.add(CtSlot(0, 14), CtSlot(0, 3));
  mapping.add(CtSlot(0, 14), CtSlot(0, 4));
  mapping.add(CtSlot(0, 14), CtSlot(0, 5));
  mapping.add(CtSlot(0, 14), CtSlot(0, 6));
  mapping.add(CtSlot(0, 14), CtSlot(0, 7));
  mapping.add(CtSlot(0, 15), CtSlot(0, 8));
  mapping.add(CtSlot(0, 15), CtSlot(0, 9));
  mapping.add(CtSlot(0, 15), CtSlot(0, 10));
  mapping.add(CtSlot(0, 15), CtSlot(0, 11));
  mapping.add(CtSlot(0, 15), CtSlot(0, 12));
  mapping.add(CtSlot(0, 15), CtSlot(0, 13));
  mapping.add(CtSlot(0, 15), CtSlot(0, 14));
  mapping.add(CtSlot(0, 15), CtSlot(0, 15));
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

TEST(ImplementShiftNetworkTest, TestTwoReplicationAlternateShiftOrder) {
  int64_t numCts = 1;
  int64_t ctSize = 16;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 14), CtSlot(0, 0));
  mapping.add(CtSlot(0, 14), CtSlot(0, 1));
  mapping.add(CtSlot(0, 14), CtSlot(0, 2));
  mapping.add(CtSlot(0, 14), CtSlot(0, 3));
  mapping.add(CtSlot(0, 14), CtSlot(0, 4));
  mapping.add(CtSlot(0, 14), CtSlot(0, 5));
  mapping.add(CtSlot(0, 14), CtSlot(0, 6));
  mapping.add(CtSlot(0, 14), CtSlot(0, 7));
  mapping.add(CtSlot(0, 15), CtSlot(0, 8));
  mapping.add(CtSlot(0, 15), CtSlot(0, 9));
  mapping.add(CtSlot(0, 15), CtSlot(0, 10));
  mapping.add(CtSlot(0, 15), CtSlot(0, 11));
  mapping.add(CtSlot(0, 15), CtSlot(0, 12));
  mapping.add(CtSlot(0, 15), CtSlot(0, 13));
  mapping.add(CtSlot(0, 15), CtSlot(0, 14));
  mapping.add(CtSlot(0, 15), CtSlot(0, 15));
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

TEST(ImplementShiftNetworkTest, TestSwapTwoCiphertexts) {
  int64_t numCts = 2;
  int64_t ctSize = 4;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(1, 0));
  mapping.add(CtSlot(0, 1), CtSlot(1, 1));
  mapping.add(CtSlot(0, 2), CtSlot(1, 2));
  mapping.add(CtSlot(0, 3), CtSlot(1, 3));
  mapping.add(CtSlot(1, 0), CtSlot(0, 0));
  mapping.add(CtSlot(1, 1), CtSlot(0, 1));
  mapping.add(CtSlot(1, 2), CtSlot(0, 2));
  mapping.add(CtSlot(1, 3), CtSlot(0, 3));
  VosVosErkinShiftNetworks shiftNetworks;
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

TEST(ImplementShiftNetworkTest, TestReorderThreeCiphertexts) {
  int64_t numCts = 3;
  int64_t ctSize = 4;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(2, 0));
  mapping.add(CtSlot(0, 1), CtSlot(2, 1));
  mapping.add(CtSlot(0, 2), CtSlot(2, 2));
  mapping.add(CtSlot(0, 3), CtSlot(2, 3));
  mapping.add(CtSlot(1, 0), CtSlot(0, 0));
  mapping.add(CtSlot(1, 1), CtSlot(0, 1));
  mapping.add(CtSlot(1, 2), CtSlot(0, 2));
  mapping.add(CtSlot(1, 3), CtSlot(0, 3));
  mapping.add(CtSlot(2, 0), CtSlot(1, 0));
  mapping.add(CtSlot(2, 1), CtSlot(1, 1));
  mapping.add(CtSlot(2, 2), CtSlot(1, 2));
  mapping.add(CtSlot(2, 3), CtSlot(1, 3));
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

TEST(ImplementShiftNetworkTest, TestSingleRotSplit) {
  int64_t numCts = 3;
  int64_t ctSize = 4;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(0, 0), CtSlot(0, 1));
  mapping.add(CtSlot(0, 1), CtSlot(0, 2));
  mapping.add(CtSlot(0, 2), CtSlot(0, 3));
  mapping.add(CtSlot(0, 3), CtSlot(1, 0));
  mapping.add(CtSlot(1, 0), CtSlot(1, 1));
  mapping.add(CtSlot(1, 1), CtSlot(1, 2));
  mapping.add(CtSlot(1, 2), CtSlot(1, 3));
  mapping.add(CtSlot(1, 3), CtSlot(2, 0));
  mapping.add(CtSlot(2, 0), CtSlot(2, 1));
  mapping.add(CtSlot(2, 1), CtSlot(2, 2));
  mapping.add(CtSlot(2, 2), CtSlot(2, 3));
  mapping.add(CtSlot(2, 3), CtSlot(0, 0));
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

TEST(ImplementShiftNetworkTest, TestRANDOM_61) {
  int64_t numCts = 24;
  int64_t ctSize = 8;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(21, 7), CtSlot(14, 2));
  mapping.add(CtSlot(23, 4), CtSlot(14, 3));
  mapping.add(CtSlot(19, 7), CtSlot(14, 4));
  mapping.add(CtSlot(19, 6), CtSlot(17, 7));
  mapping.add(CtSlot(21, 7), CtSlot(18, 3));
  mapping.add(CtSlot(19, 3), CtSlot(18, 7));  // This is the problematic entry
  mapping.add(CtSlot(12, 6), CtSlot(20, 6));
  mapping.add(CtSlot(19, 2), CtSlot(20, 7));
  mapping.add(CtSlot(19, 1), CtSlot(22, 0));
  mapping.add(CtSlot(14, 7), CtSlot(22, 2));
  mapping.add(CtSlot(16, 5), CtSlot(22, 4));
  mapping.add(CtSlot(19, 6), CtSlot(23, 2));
  mapping.add(CtSlot(16, 6), CtSlot(23, 3));
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

TEST(ImplementShiftNetworkTest, TestRANDOM_62) {
  int64_t numCts = 24;
  int64_t ctSize = 8;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(23, 4), CtSlot(0, 0));
  mapping.add(CtSlot(16, 7), CtSlot(0, 1));
  mapping.add(CtSlot(1, 1), CtSlot(0, 2));
  mapping.add(CtSlot(0, 0), CtSlot(0, 3));
  mapping.add(CtSlot(8, 4), CtSlot(0, 4));
  mapping.add(CtSlot(9, 0), CtSlot(0, 5));
  mapping.add(CtSlot(10, 7), CtSlot(0, 6));
  mapping.add(CtSlot(23, 1), CtSlot(0, 7));
  mapping.add(CtSlot(17, 5), CtSlot(1, 0));
  mapping.add(CtSlot(21, 4), CtSlot(1, 1));
  mapping.add(CtSlot(17, 5), CtSlot(1, 2));
  mapping.add(CtSlot(5, 1), CtSlot(1, 3));
  mapping.add(CtSlot(2, 7), CtSlot(1, 4));
  mapping.add(CtSlot(6, 6), CtSlot(1, 5));
  mapping.add(CtSlot(17, 1), CtSlot(1, 6));
  mapping.add(CtSlot(19, 4), CtSlot(1, 7));
  mapping.add(CtSlot(14, 7), CtSlot(2, 0));
  mapping.add(CtSlot(10, 4), CtSlot(2, 1));
  mapping.add(CtSlot(0, 0), CtSlot(2, 2));
  mapping.add(CtSlot(18, 4), CtSlot(2, 3));
  mapping.add(CtSlot(16, 4), CtSlot(2, 4));
  mapping.add(CtSlot(4, 4), CtSlot(2, 5));
  mapping.add(CtSlot(16, 7), CtSlot(2, 6));
  mapping.add(CtSlot(9, 3), CtSlot(2, 7));
  mapping.add(CtSlot(0, 6), CtSlot(3, 0));
  mapping.add(CtSlot(23, 0), CtSlot(3, 1));
  mapping.add(CtSlot(5, 5), CtSlot(3, 2));
  mapping.add(CtSlot(20, 3), CtSlot(3, 3));
  mapping.add(CtSlot(12, 1), CtSlot(3, 4));
  mapping.add(CtSlot(6, 7), CtSlot(3, 5));
  mapping.add(CtSlot(14, 0), CtSlot(3, 6));
  mapping.add(CtSlot(9, 4), CtSlot(3, 7));
  mapping.add(CtSlot(2, 4), CtSlot(4, 0));
  mapping.add(CtSlot(10, 0), CtSlot(4, 1));
  mapping.add(CtSlot(13, 5), CtSlot(4, 2));
  mapping.add(CtSlot(23, 5), CtSlot(4, 3));
  mapping.add(CtSlot(10, 1), CtSlot(4, 4));
  mapping.add(CtSlot(3, 7), CtSlot(4, 5));
  mapping.add(CtSlot(0, 5), CtSlot(4, 6));
  mapping.add(CtSlot(4, 5), CtSlot(4, 7));
  mapping.add(CtSlot(6, 6), CtSlot(5, 0));
  mapping.add(CtSlot(6, 6), CtSlot(5, 1));
  mapping.add(CtSlot(12, 5), CtSlot(5, 2));
  mapping.add(CtSlot(15, 0), CtSlot(5, 3));
  mapping.add(CtSlot(5, 6), CtSlot(5, 4));
  mapping.add(CtSlot(3, 0), CtSlot(5, 5));
  mapping.add(CtSlot(17, 2), CtSlot(5, 6));
  mapping.add(CtSlot(7, 5), CtSlot(5, 7));
  mapping.add(CtSlot(1, 1), CtSlot(6, 0));
  mapping.add(CtSlot(22, 5), CtSlot(6, 1));
  mapping.add(CtSlot(13, 7), CtSlot(6, 2));
  mapping.add(CtSlot(11, 4), CtSlot(6, 3));
  mapping.add(CtSlot(22, 1), CtSlot(6, 4));
  mapping.add(CtSlot(3, 5), CtSlot(6, 5));
  mapping.add(CtSlot(7, 7), CtSlot(6, 6));
  mapping.add(CtSlot(2, 1), CtSlot(6, 7));
  mapping.add(CtSlot(16, 1), CtSlot(7, 0));
  mapping.add(CtSlot(3, 7), CtSlot(7, 1));
  mapping.add(CtSlot(8, 0), CtSlot(7, 2));
  mapping.add(CtSlot(17, 6), CtSlot(7, 3));
  mapping.add(CtSlot(14, 5), CtSlot(7, 4));
  mapping.add(CtSlot(0, 3), CtSlot(7, 5));
  mapping.add(CtSlot(23, 0), CtSlot(7, 6));
  mapping.add(CtSlot(7, 6), CtSlot(7, 7));
  mapping.add(CtSlot(20, 7), CtSlot(8, 0));
  mapping.add(CtSlot(0, 2), CtSlot(8, 1));
  mapping.add(CtSlot(5, 6), CtSlot(8, 2));
  mapping.add(CtSlot(6, 7), CtSlot(8, 3));
  mapping.add(CtSlot(9, 7), CtSlot(8, 4));
  mapping.add(CtSlot(18, 7), CtSlot(8, 5));
  mapping.add(CtSlot(1, 1), CtSlot(8, 6));
  mapping.add(CtSlot(7, 7), CtSlot(8, 7));
  mapping.add(CtSlot(12, 7), CtSlot(9, 0));
  mapping.add(CtSlot(7, 5), CtSlot(9, 1));
  mapping.add(CtSlot(4, 2), CtSlot(9, 2));
  mapping.add(CtSlot(2, 0), CtSlot(9, 3));
  mapping.add(CtSlot(13, 5), CtSlot(9, 4));
  mapping.add(CtSlot(19, 0), CtSlot(9, 5));
  mapping.add(CtSlot(0, 0), CtSlot(9, 6));
  mapping.add(CtSlot(2, 4), CtSlot(9, 7));
  mapping.add(CtSlot(13, 2), CtSlot(10, 0));
  mapping.add(CtSlot(11, 0), CtSlot(10, 1));
  mapping.add(CtSlot(13, 4), CtSlot(10, 2));
  mapping.add(CtSlot(16, 5), CtSlot(10, 3));
  mapping.add(CtSlot(2, 1), CtSlot(10, 4));
  mapping.add(CtSlot(4, 1), CtSlot(10, 5));
  mapping.add(CtSlot(10, 1), CtSlot(10, 6));
  mapping.add(CtSlot(11, 2), CtSlot(10, 7));
  mapping.add(CtSlot(8, 0), CtSlot(11, 0));
  mapping.add(CtSlot(21, 0), CtSlot(11, 1));
  mapping.add(CtSlot(10, 2), CtSlot(11, 2));
  mapping.add(CtSlot(16, 0), CtSlot(11, 3));
  mapping.add(CtSlot(5, 7), CtSlot(11, 4));
  mapping.add(CtSlot(23, 6), CtSlot(11, 5));
  mapping.add(CtSlot(4, 6), CtSlot(11, 6));
  mapping.add(CtSlot(6, 3), CtSlot(11, 7));
  mapping.add(CtSlot(22, 1), CtSlot(12, 0));
  mapping.add(CtSlot(1, 5), CtSlot(12, 1));
  mapping.add(CtSlot(2, 4), CtSlot(12, 2));
  mapping.add(CtSlot(4, 3), CtSlot(12, 3));
  mapping.add(CtSlot(8, 3), CtSlot(12, 4));
  mapping.add(CtSlot(16, 4), CtSlot(12, 5));
  mapping.add(CtSlot(23, 5), CtSlot(12, 6));
  mapping.add(CtSlot(19, 0), CtSlot(12, 7));
  mapping.add(CtSlot(11, 7), CtSlot(13, 0));
  mapping.add(CtSlot(13, 4), CtSlot(13, 1));
  mapping.add(CtSlot(12, 1), CtSlot(13, 2));
  mapping.add(CtSlot(2, 3), CtSlot(13, 3));
  mapping.add(CtSlot(21, 5), CtSlot(13, 4));
  mapping.add(CtSlot(4, 6), CtSlot(13, 5));
  mapping.add(CtSlot(16, 5), CtSlot(13, 6));
  mapping.add(CtSlot(12, 6), CtSlot(13, 7));
  mapping.add(CtSlot(2, 7), CtSlot(14, 0));
  mapping.add(CtSlot(2, 3), CtSlot(14, 1));
  mapping.add(CtSlot(21, 7), CtSlot(14, 2));
  mapping.add(CtSlot(23, 4), CtSlot(14, 3));
  mapping.add(CtSlot(19, 7), CtSlot(14, 4));
  mapping.add(CtSlot(0, 1), CtSlot(14, 5));
  mapping.add(CtSlot(15, 4), CtSlot(14, 6));
  mapping.add(CtSlot(0, 6), CtSlot(14, 7));
  mapping.add(CtSlot(11, 0), CtSlot(15, 0));
  mapping.add(CtSlot(17, 7), CtSlot(15, 1));
  mapping.add(CtSlot(18, 0), CtSlot(15, 2));
  mapping.add(CtSlot(15, 0), CtSlot(15, 3));
  mapping.add(CtSlot(4, 1), CtSlot(15, 4));
  mapping.add(CtSlot(12, 6), CtSlot(15, 5));
  mapping.add(CtSlot(12, 5), CtSlot(15, 6));
  mapping.add(CtSlot(5, 2), CtSlot(15, 7));
  mapping.add(CtSlot(2, 5), CtSlot(16, 0));
  mapping.add(CtSlot(23, 0), CtSlot(16, 1));
  mapping.add(CtSlot(10, 2), CtSlot(16, 2));
  mapping.add(CtSlot(9, 7), CtSlot(16, 3));
  mapping.add(CtSlot(2, 6), CtSlot(16, 4));
  mapping.add(CtSlot(7, 0), CtSlot(16, 5));
  mapping.add(CtSlot(20, 2), CtSlot(16, 6));
  mapping.add(CtSlot(13, 0), CtSlot(16, 7));
  mapping.add(CtSlot(7, 4), CtSlot(17, 0));
  mapping.add(CtSlot(6, 4), CtSlot(17, 1));
  mapping.add(CtSlot(12, 1), CtSlot(17, 2));
  mapping.add(CtSlot(5, 5), CtSlot(17, 3));
  mapping.add(CtSlot(8, 4), CtSlot(17, 4));
  mapping.add(CtSlot(14, 7), CtSlot(17, 5));
  mapping.add(CtSlot(18, 3), CtSlot(17, 6));
  mapping.add(CtSlot(19, 6), CtSlot(17, 7));
  mapping.add(CtSlot(16, 2), CtSlot(18, 0));
  mapping.add(CtSlot(14, 7), CtSlot(18, 1));
  mapping.add(CtSlot(3, 3), CtSlot(18, 2));
  mapping.add(CtSlot(21, 7), CtSlot(18, 3));
  mapping.add(CtSlot(17, 6), CtSlot(18, 4));
  mapping.add(CtSlot(5, 4), CtSlot(18, 5));
  mapping.add(CtSlot(8, 2), CtSlot(18, 6));
  mapping.add(CtSlot(19, 3), CtSlot(18, 7));
  mapping.add(CtSlot(11, 7), CtSlot(19, 0));
  mapping.add(CtSlot(8, 5), CtSlot(19, 1));
  mapping.add(CtSlot(17, 4), CtSlot(19, 2));
  mapping.add(CtSlot(6, 6), CtSlot(19, 3));
  mapping.add(CtSlot(14, 0), CtSlot(19, 4));
  mapping.add(CtSlot(20, 5), CtSlot(19, 5));
  mapping.add(CtSlot(0, 1), CtSlot(19, 6));
  mapping.add(CtSlot(12, 4), CtSlot(19, 7));
  mapping.add(CtSlot(13, 5), CtSlot(20, 0));
  mapping.add(CtSlot(20, 4), CtSlot(20, 1));
  mapping.add(CtSlot(11, 1), CtSlot(20, 2));
  mapping.add(CtSlot(2, 6), CtSlot(20, 3));
  mapping.add(CtSlot(7, 7), CtSlot(20, 4));
  mapping.add(CtSlot(6, 1), CtSlot(20, 5));
  mapping.add(CtSlot(12, 6), CtSlot(20, 6));
  mapping.add(CtSlot(19, 2), CtSlot(20, 7));
  mapping.add(CtSlot(18, 7), CtSlot(21, 0));
  mapping.add(CtSlot(14, 5), CtSlot(21, 1));
  mapping.add(CtSlot(12, 5), CtSlot(21, 2));
  mapping.add(CtSlot(18, 4), CtSlot(21, 3));
  mapping.add(CtSlot(22, 5), CtSlot(21, 4));
  mapping.add(CtSlot(2, 0), CtSlot(21, 5));
  mapping.add(CtSlot(4, 7), CtSlot(21, 6));
  mapping.add(CtSlot(21, 0), CtSlot(21, 7));
  mapping.add(CtSlot(19, 1), CtSlot(22, 0));
  mapping.add(CtSlot(0, 7), CtSlot(22, 1));
  mapping.add(CtSlot(14, 7), CtSlot(22, 2));
  mapping.add(CtSlot(4, 7), CtSlot(22, 3));
  mapping.add(CtSlot(16, 5), CtSlot(22, 4));
  mapping.add(CtSlot(10, 1), CtSlot(22, 5));
  mapping.add(CtSlot(19, 2), CtSlot(22, 6));
  mapping.add(CtSlot(4, 1), CtSlot(22, 7));
  mapping.add(CtSlot(11, 1), CtSlot(23, 0));
  mapping.add(CtSlot(2, 2), CtSlot(23, 1));
  mapping.add(CtSlot(19, 6), CtSlot(23, 2));
  mapping.add(CtSlot(16, 6), CtSlot(23, 3));
  mapping.add(CtSlot(10, 1), CtSlot(23, 4));
  mapping.add(CtSlot(10, 1), CtSlot(23, 5));
  mapping.add(CtSlot(23, 5), CtSlot(23, 6));
  mapping.add(CtSlot(5, 1), CtSlot(23, 7));
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

TEST(ImplementShiftNetworkTest, TestRANDOM_64) {
  int64_t numCts = 24;
  int64_t ctSize = 8;
  Mapping mapping(ctSize, numCts);
  mapping.add(CtSlot(23, 4), CtSlot(0, 0));
  mapping.add(CtSlot(16, 7), CtSlot(0, 1));
  mapping.add(CtSlot(1, 1), CtSlot(0, 2));
  mapping.add(CtSlot(0, 0), CtSlot(0, 3));
  mapping.add(CtSlot(8, 4), CtSlot(0, 4));
  mapping.add(CtSlot(9, 0), CtSlot(0, 5));
  mapping.add(CtSlot(10, 7), CtSlot(0, 6));
  mapping.add(CtSlot(23, 1), CtSlot(0, 7));
  mapping.add(CtSlot(17, 5), CtSlot(1, 0));
  mapping.add(CtSlot(21, 4), CtSlot(1, 1));
  mapping.add(CtSlot(17, 5), CtSlot(1, 2));
  mapping.add(CtSlot(5, 1), CtSlot(1, 3));
  mapping.add(CtSlot(2, 7), CtSlot(1, 4));
  mapping.add(CtSlot(6, 6), CtSlot(1, 5));
  mapping.add(CtSlot(17, 1), CtSlot(1, 6));
  mapping.add(CtSlot(19, 4), CtSlot(1, 7));
  mapping.add(CtSlot(14, 7), CtSlot(2, 0));
  mapping.add(CtSlot(10, 4), CtSlot(2, 1));
  mapping.add(CtSlot(0, 0), CtSlot(2, 2));
  mapping.add(CtSlot(18, 4), CtSlot(2, 3));
  mapping.add(CtSlot(16, 4), CtSlot(2, 4));
  mapping.add(CtSlot(4, 4), CtSlot(2, 5));
  mapping.add(CtSlot(16, 7), CtSlot(2, 6));
  mapping.add(CtSlot(9, 3), CtSlot(2, 7));
  mapping.add(CtSlot(0, 6), CtSlot(3, 0));
  mapping.add(CtSlot(23, 0), CtSlot(3, 1));
  mapping.add(CtSlot(5, 5), CtSlot(3, 2));
  mapping.add(CtSlot(20, 3), CtSlot(3, 3));
  mapping.add(CtSlot(12, 1), CtSlot(3, 4));
  mapping.add(CtSlot(6, 7), CtSlot(3, 5));
  mapping.add(CtSlot(14, 0), CtSlot(3, 6));
  mapping.add(CtSlot(9, 4), CtSlot(3, 7));
  mapping.add(CtSlot(2, 4), CtSlot(4, 0));
  mapping.add(CtSlot(10, 0), CtSlot(4, 1));
  mapping.add(CtSlot(13, 5), CtSlot(4, 2));
  mapping.add(CtSlot(23, 5), CtSlot(4, 3));
  mapping.add(CtSlot(10, 1), CtSlot(4, 4));
  mapping.add(CtSlot(3, 7), CtSlot(4, 5));
  mapping.add(CtSlot(0, 5), CtSlot(4, 6));
  mapping.add(CtSlot(4, 5), CtSlot(4, 7));
  mapping.add(CtSlot(6, 6), CtSlot(5, 0));
  mapping.add(CtSlot(6, 6), CtSlot(5, 1));
  mapping.add(CtSlot(12, 5), CtSlot(5, 2));
  mapping.add(CtSlot(15, 0), CtSlot(5, 3));
  mapping.add(CtSlot(5, 6), CtSlot(5, 4));
  mapping.add(CtSlot(3, 0), CtSlot(5, 5));
  mapping.add(CtSlot(17, 2), CtSlot(5, 6));
  mapping.add(CtSlot(7, 5), CtSlot(5, 7));
  mapping.add(CtSlot(1, 1), CtSlot(6, 0));
  mapping.add(CtSlot(22, 5), CtSlot(6, 1));
  mapping.add(CtSlot(13, 7), CtSlot(6, 2));
  mapping.add(CtSlot(11, 4), CtSlot(6, 3));
  mapping.add(CtSlot(22, 1), CtSlot(6, 4));
  mapping.add(CtSlot(3, 5), CtSlot(6, 5));
  mapping.add(CtSlot(7, 7), CtSlot(6, 6));
  mapping.add(CtSlot(2, 1), CtSlot(6, 7));
  mapping.add(CtSlot(16, 1), CtSlot(7, 0));
  mapping.add(CtSlot(3, 7), CtSlot(7, 1));
  mapping.add(CtSlot(8, 0), CtSlot(7, 2));
  mapping.add(CtSlot(17, 1), CtSlot(7, 3));
  mapping.add(CtSlot(14, 5), CtSlot(7, 4));
  mapping.add(CtSlot(0, 3), CtSlot(7, 5));
  mapping.add(CtSlot(23, 0), CtSlot(7, 6));
  mapping.add(CtSlot(7, 6), CtSlot(7, 7));
  mapping.add(CtSlot(20, 7), CtSlot(8, 0));
  mapping.add(CtSlot(0, 0), CtSlot(8, 1));
  mapping.add(CtSlot(5, 6), CtSlot(8, 2));
  mapping.add(CtSlot(6, 7), CtSlot(8, 3));
  mapping.add(CtSlot(9, 7), CtSlot(8, 4));
  mapping.add(CtSlot(18, 7), CtSlot(8, 5));
  mapping.add(CtSlot(1, 1), CtSlot(8, 6));
  mapping.add(CtSlot(7, 7), CtSlot(8, 7));
  mapping.add(CtSlot(12, 7), CtSlot(9, 0));
  mapping.add(CtSlot(7, 5), CtSlot(9, 1));
  mapping.add(CtSlot(4, 2), CtSlot(9, 2));
  mapping.add(CtSlot(2, 0), CtSlot(9, 3));
  mapping.add(CtSlot(13, 5), CtSlot(9, 4));
  mapping.add(CtSlot(19, 0), CtSlot(9, 5));
  mapping.add(CtSlot(0, 0), CtSlot(9, 6));
  mapping.add(CtSlot(2, 4), CtSlot(9, 7));
  mapping.add(CtSlot(13, 2), CtSlot(10, 0));
  mapping.add(CtSlot(11, 0), CtSlot(10, 1));
  mapping.add(CtSlot(13, 4), CtSlot(10, 2));
  mapping.add(CtSlot(16, 5), CtSlot(10, 3));
  mapping.add(CtSlot(2, 1), CtSlot(10, 4));
  mapping.add(CtSlot(4, 1), CtSlot(10, 5));
  mapping.add(CtSlot(10, 1), CtSlot(10, 6));
  mapping.add(CtSlot(11, 2), CtSlot(10, 7));
  mapping.add(CtSlot(8, 0), CtSlot(11, 0));
  mapping.add(CtSlot(21, 0), CtSlot(11, 1));
  mapping.add(CtSlot(10, 2), CtSlot(11, 2));
  mapping.add(CtSlot(16, 0), CtSlot(11, 3));
  mapping.add(CtSlot(5, 7), CtSlot(11, 4));
  mapping.add(CtSlot(23, 6), CtSlot(11, 5));
  mapping.add(CtSlot(4, 6), CtSlot(11, 6));
  mapping.add(CtSlot(6, 3), CtSlot(11, 7));
  mapping.add(CtSlot(22, 1), CtSlot(12, 0));
  mapping.add(CtSlot(1, 5), CtSlot(12, 1));
  mapping.add(CtSlot(2, 4), CtSlot(12, 2));
  mapping.add(CtSlot(4, 3), CtSlot(12, 3));
  mapping.add(CtSlot(8, 3), CtSlot(12, 4));
  mapping.add(CtSlot(16, 4), CtSlot(12, 5));
  mapping.add(CtSlot(23, 5), CtSlot(12, 6));
  mapping.add(CtSlot(19, 0), CtSlot(12, 7));
  mapping.add(CtSlot(11, 7), CtSlot(13, 0));
  mapping.add(CtSlot(13, 4), CtSlot(13, 1));
  mapping.add(CtSlot(12, 1), CtSlot(13, 2));
  mapping.add(CtSlot(2, 3), CtSlot(13, 3));
  mapping.add(CtSlot(21, 5), CtSlot(13, 4));
  mapping.add(CtSlot(4, 6), CtSlot(13, 5));
  mapping.add(CtSlot(16, 5), CtSlot(13, 6));
  mapping.add(CtSlot(12, 6), CtSlot(13, 7));
  mapping.add(CtSlot(2, 7), CtSlot(14, 0));
  mapping.add(CtSlot(2, 3), CtSlot(14, 1));
  mapping.add(CtSlot(21, 7), CtSlot(14, 2));
  mapping.add(CtSlot(23, 4), CtSlot(14, 3));
  mapping.add(CtSlot(19, 7), CtSlot(14, 4));
  mapping.add(CtSlot(0, 1), CtSlot(14, 5));
  mapping.add(CtSlot(15, 4), CtSlot(14, 6));
  mapping.add(CtSlot(0, 6), CtSlot(14, 7));
  mapping.add(CtSlot(11, 0), CtSlot(15, 0));
  mapping.add(CtSlot(17, 7), CtSlot(15, 1));
  mapping.add(CtSlot(18, 0), CtSlot(15, 2));
  mapping.add(CtSlot(15, 0), CtSlot(15, 3));
  mapping.add(CtSlot(4, 1), CtSlot(15, 4));
  mapping.add(CtSlot(12, 6), CtSlot(15, 5));
  mapping.add(CtSlot(12, 5), CtSlot(15, 6));
  mapping.add(CtSlot(5, 2), CtSlot(15, 7));
  mapping.add(CtSlot(2, 5), CtSlot(16, 0));
  mapping.add(CtSlot(23, 0), CtSlot(16, 1));
  mapping.add(CtSlot(10, 2), CtSlot(16, 2));
  mapping.add(CtSlot(9, 7), CtSlot(16, 3));
  mapping.add(CtSlot(2, 6), CtSlot(16, 4));
  mapping.add(CtSlot(7, 0), CtSlot(16, 5));
  mapping.add(CtSlot(20, 2), CtSlot(16, 6));
  mapping.add(CtSlot(13, 0), CtSlot(16, 7));
  mapping.add(CtSlot(7, 4), CtSlot(17, 0));
  mapping.add(CtSlot(6, 4), CtSlot(17, 1));
  mapping.add(CtSlot(12, 1), CtSlot(17, 2));
  mapping.add(CtSlot(5, 5), CtSlot(17, 3));
  mapping.add(CtSlot(8, 4), CtSlot(17, 4));
  mapping.add(CtSlot(14, 7), CtSlot(17, 5));
  mapping.add(CtSlot(18, 3), CtSlot(17, 6));
  mapping.add(CtSlot(19, 6), CtSlot(17, 7));
  mapping.add(CtSlot(16, 2), CtSlot(18, 0));
  mapping.add(CtSlot(14, 7), CtSlot(18, 1));
  mapping.add(CtSlot(3, 3), CtSlot(18, 2));
  mapping.add(CtSlot(21, 7), CtSlot(18, 3));
  mapping.add(CtSlot(17, 6), CtSlot(18, 4));
  mapping.add(CtSlot(5, 4), CtSlot(18, 5));
  mapping.add(CtSlot(8, 2), CtSlot(18, 6));
  mapping.add(CtSlot(19, 3), CtSlot(18, 7));
  mapping.add(CtSlot(11, 7), CtSlot(19, 0));
  mapping.add(CtSlot(8, 5), CtSlot(19, 1));
  mapping.add(CtSlot(17, 4), CtSlot(19, 2));
  mapping.add(CtSlot(6, 6), CtSlot(19, 3));
  mapping.add(CtSlot(14, 0), CtSlot(19, 4));
  mapping.add(CtSlot(20, 5), CtSlot(19, 5));
  mapping.add(CtSlot(0, 1), CtSlot(19, 6));
  mapping.add(CtSlot(12, 4), CtSlot(19, 7));
  mapping.add(CtSlot(13, 5), CtSlot(20, 0));
  mapping.add(CtSlot(20, 4), CtSlot(20, 1));
  mapping.add(CtSlot(11, 1), CtSlot(20, 2));
  mapping.add(CtSlot(2, 6), CtSlot(20, 3));
  mapping.add(CtSlot(7, 7), CtSlot(20, 4));
  mapping.add(CtSlot(6, 1), CtSlot(20, 5));
  mapping.add(CtSlot(12, 6), CtSlot(20, 6));
  mapping.add(CtSlot(19, 2), CtSlot(20, 7));
  mapping.add(CtSlot(18, 7), CtSlot(21, 0));
  mapping.add(CtSlot(14, 5), CtSlot(21, 1));
  mapping.add(CtSlot(12, 5), CtSlot(21, 2));
  mapping.add(CtSlot(18, 4), CtSlot(21, 3));
  mapping.add(CtSlot(22, 5), CtSlot(21, 4));
  mapping.add(CtSlot(2, 0), CtSlot(21, 5));
  mapping.add(CtSlot(4, 7), CtSlot(21, 6));
  mapping.add(CtSlot(21, 0), CtSlot(21, 7));
  mapping.add(CtSlot(19, 1), CtSlot(22, 0));
  mapping.add(CtSlot(0, 7), CtSlot(22, 1));
  mapping.add(CtSlot(14, 7), CtSlot(22, 2));
  mapping.add(CtSlot(4, 7), CtSlot(22, 3));
  mapping.add(CtSlot(16, 5), CtSlot(22, 4));
  mapping.add(CtSlot(10, 1), CtSlot(22, 5));
  mapping.add(CtSlot(19, 2), CtSlot(22, 6));
  mapping.add(CtSlot(4, 1), CtSlot(22, 7));
  mapping.add(CtSlot(11, 1), CtSlot(23, 0));
  mapping.add(CtSlot(2, 2), CtSlot(23, 1));
  mapping.add(CtSlot(19, 6), CtSlot(23, 2));
  mapping.add(CtSlot(16, 6), CtSlot(23, 3));
  mapping.add(CtSlot(10, 1), CtSlot(23, 4));
  mapping.add(CtSlot(10, 1), CtSlot(23, 5));
  mapping.add(CtSlot(23, 5), CtSlot(23, 6));
  mapping.add(CtSlot(5, 1), CtSlot(23, 7));
  EXPECT_TRUE(checkMapping(mapping, numCts, ctSize));
}

}  // namespace
}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
