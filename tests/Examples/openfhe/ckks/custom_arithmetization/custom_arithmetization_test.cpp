#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/custom_arithmetization/custom_arithmetization_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(CustomArithmetizationTest, RunTest) {
  auto cryptoContext = matmul__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = matmul__configure_crypto_context(cryptoContext, secretKey);

  // ct is a [4,4] input matrix that is packed column-wise
  // [
  //  [0 1 2 3]
  //  [4 5 6 7]
  //  [8 9 10 11]
  //  [12 13 14 15]
  // ]
  // =>
  // [0 4 8 12 1 5 9 13 2 6 10 14 3 7 11 15]
  std::vector<float> m;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      m.push_back((float)(j * 4 + i));
    }
  }

  // pt is a [4,4] input matrix that is packed in repeated diagonals
  // [
  //  [0 1 2 3]
  //  [4 5 6 7]
  //  [8 9 10 11]
  //  [12 13 14 15]
  // ]
  // =>
  // d0:[0 0 0 0 5 5 5 5 10 10 10 10 15 15 15 15]
  // d1:[4 4 4 4 9 9 9 9 13 13 13 13 3 3 3 3 3]
  std::vector<float> d0 = {0.0,  0.0,  0.0,  0.0,  5.0,  5.0,  5.0,  5.0,
                           10.0, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0, 15.0};
  std::vector<float> d1 = {4.0,  4.0,  4.0,  4.0,  9.0, 9.0, 9.0, 9.0,
                           14.0, 14.0, 14.0, 14.0, 3.0, 3.0, 3.0, 3.0};

  // rotated by 8 for baby-step giant-step
  // d2:[2 2 2 2 7 7 7 7 8 8 8 8 13 13 13 13]
  // d3:[6 6 6 6 11 11 11 11 12 12 12 12 1 1 1 1]
  std::vector<float> d2 = {2.0, 2.0, 2.0, 2.0, 7.0,  7.0,  7.0,  7.0,
                           8.0, 8.0, 8.0, 8.0, 13.0, 13.0, 13.0, 13.0};
  std::vector<float> d3 = {6.0,  6.0,  6.0,  6.0,  11.0, 11.0, 11.0, 11.0,
                           12.0, 12.0, 12.0, 12.0, 1.0,  1.0,  1.0,  1.0};

  // expected is the result of the matrix multiplication packed column-wise
  // [
  //  [ 56  62  68  74]
  //  [152 174 196 218]
  //  [248 286 324 362]
  //  [344 398 452 506]
  // ]
  std::vector<float> expected = {56.0,  152.0, 248.0, 344.0, 62.0,  174.0,
                                 286.0, 398.0, 68.0,  196.0, 324.0, 452.0,
                                 74.0,  218.0, 362.0, 506.0};

  auto ctEncrypted = matmul__encrypt__arg0(cryptoContext, m, keyPair.publicKey);

  auto result = matmul(cryptoContext, ctEncrypted, d0, d1, d2, d3);

  auto actual =
      matmul__decrypt__result0(cryptoContext, result, keyPair.secretKey);

  ASSERT_EQ(actual.size(), expected.size());

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[i], 1e-3);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
