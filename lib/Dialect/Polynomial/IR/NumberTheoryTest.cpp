#include <cstddef>
#include <unordered_set>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Polynomial/IR/NumberTheory.h"
#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

namespace mlir::heir::polynomial {
namespace {

TEST(NumberTheoryTest, TestIsPrimitive2nthRootOfUnity) {
  auto root = llvm::APInt(64, 3);
  auto n = 5;  // checking 2n = 10
  auto cmod = llvm::APInt(64, 11);
  bool expected = false;
  EXPECT_EQ(expected, isPrimitiveNthRootOfUnity(root, 2 * n, cmod));
}

TEST(NumberTheoryTest, TestIsPrimitiveNthRootOfUnity) {
  auto root = llvm::APInt(64, 3);
  auto n = 5;
  auto cmod = llvm::APInt(64, 11);
  bool expected = true;
  EXPECT_EQ(expected, isPrimitiveNthRootOfUnity(root, n, cmod));
}

TEST(NumberTheoryTest, TestAllRootsMod23) {
  auto cmod = llvm::APInt(64, 23);
  auto base = llvm::APInt(64, 5);
  auto root = llvm::APInt(64, 1);  // 5^0
  std::unordered_set<unsigned> powersOf5 = {1, 3, 5, 7, 9, 13, 15, 17, 19, 21};
  for (size_t i = 1; i < 23; ++i) {
    root = (root * base).urem(cmod);
    bool expected = powersOf5.count(i) > 0;
    EXPECT_EQ(expected, isPrimitiveNthRootOfUnity(root, 22, cmod));
  }
}

TEST(NumberTheoryTest, NonRootsMod171) {
  auto cmod = llvm::APInt(64, 171);
  for (size_t i = 1; i < 171; ++i) {
    auto root = llvm::APInt(64, i);
    EXPECT_EQ(false, isPrimitiveNthRootOfUnity(root, 170, cmod));
  }
}

TEST(NumberTheoryTest, CountRootsMod173) {
  unsigned count = 0;
  auto cmod = llvm::APInt(64, 173);
  for (size_t i = 1; i < 173; ++i) {
    auto root = llvm::APInt(64, i);
    if (isPrimitiveNthRootOfUnity(root, 172, cmod)) {
      count++;
    }
  }
  EXPECT_EQ(84, count);
}

}  // namespace
}  // namespace mlir::heir::polynomial
