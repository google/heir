#include <cstdint>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"                        // from @googletest
#include "src/binfhe/include/binfhecontext.h"   // from @openfhe
#include "src/binfhe/include/lwe-ciphertext.h"  // from @openfhe
#include "src/binfhe/include/lwe-privatekey.h"  // from @openfhe
#include "src/binfhe/include/lwe-publickey.h"   // from @openfhe

// Generated header (will be created by the build system)
#include "tests/Examples/openfhe/binfhe/xor_bits_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

class BinFHELogicTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize BinFHE context with default parameters
    using namespace lbcrypto;

    // Create BinFHE context with standard security parameters
    auto cc = BinFHEContext();
    // For arbitrary function evaluation (EvalFunc), BinFHE requires q <= ring
    // dimension. Use the arbFunc=true overload as shown in OpenFHE's
    // eval-function example. logQ=12 is a standard setting for arbitrary
    // function evaluation.
    cc.GenerateBinFHEContext(STD128, /*arbFunc=*/true, /*logQ=*/12, /*N=*/0,
                             GINX, /*timeOptimization=*/false);

    // Generate keys
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);

    cryptoContext = std::make_shared<BinFHEContext>(cc);
    secretKey = sk;
  }

  BinFHEContextT cryptoContext;
  lbcrypto::LWEPrivateKey secretKey;
};

TEST_F(BinFHELogicTest, TestXOR) {
  using namespace lbcrypto;

  // Test all XOR combinations
  std::vector<std::pair<int, int>> test_cases = {
      {0, 0},  // 0 XOR 0 = 0
      {0, 1},  // 0 XOR 1 = 1
      {1, 0},  // 1 XOR 0 = 1
      {1, 1}   // 1 XOR 1 = 0
  };

  for (const auto& [a, b] : test_cases) {
    // Encrypt inputs using the ptxt_mod from the generated code (8)
    auto ct_a =
        cryptoContext->Encrypt(secretKey, a, BINFHE_OUTPUT::SMALL_DIM, 8);
    auto ct_b =
        cryptoContext->Encrypt(secretKey, b, BINFHE_OUTPUT::SMALL_DIM, 8);

    // Compute XOR
    auto ct_result = xor_bits(cryptoContext, ct_a, ct_b);

    // Decrypt result
    LWEPlaintext result;
    cryptoContext->Decrypt(secretKey, ct_result, &result, 8);

    // Check result
    int expected = (a ^ b) ? 1 : 0;
    EXPECT_EQ(result, expected) << "XOR(" << a << ", " << b << ") failed";
  }
}

TEST_F(BinFHELogicTest, TestAND) {
  using namespace lbcrypto;

  // Test all AND combinations
  std::vector<std::pair<int, int>> test_cases = {
      {0, 0},  // 0 AND 0 = 0
      {0, 1},  // 0 AND 1 = 0
      {1, 0},  // 1 AND 0 = 0
      {1, 1}   // 1 AND 1 = 1
  };

  for (const auto& [a, b] : test_cases) {
    // Encrypt inputs
    auto ct_a =
        cryptoContext->Encrypt(secretKey, a, BINFHE_OUTPUT::SMALL_DIM, 8);
    auto ct_b =
        cryptoContext->Encrypt(secretKey, b, BINFHE_OUTPUT::SMALL_DIM, 8);

    // Compute AND
    auto ct_result = and_bits(cryptoContext, ct_a, ct_b);

    // Decrypt result
    LWEPlaintext result;
    cryptoContext->Decrypt(secretKey, ct_result, &result, 8);

    // Check result
    int expected = (a & b) ? 1 : 0;
    EXPECT_EQ(result, expected) << "AND(" << a << ", " << b << ") failed";
  }
}

TEST_F(BinFHELogicTest, TestOR) {
  using namespace lbcrypto;

  // Test all OR combinations
  std::vector<std::pair<int, int>> test_cases = {
      {0, 0},  // 0 OR 0 = 0
      {0, 1},  // 0 OR 1 = 1
      {1, 0},  // 1 OR 0 = 1
      {1, 1}   // 1 OR 1 = 1
  };

  for (const auto& [a, b] : test_cases) {
    // Encrypt inputs
    auto ct_a =
        cryptoContext->Encrypt(secretKey, a, BINFHE_OUTPUT::SMALL_DIM, 8);
    auto ct_b =
        cryptoContext->Encrypt(secretKey, b, BINFHE_OUTPUT::SMALL_DIM, 8);

    // Compute OR
    auto ct_result = or_bits(cryptoContext, ct_a, ct_b);

    // Decrypt result
    LWEPlaintext result;
    cryptoContext->Decrypt(secretKey, ct_result, &result, 8);

    // Check result
    int expected = (a | b) ? 1 : 0;
    EXPECT_EQ(result, expected) << "OR(" << a << ", " << b << ") failed";
  }
}

TEST_F(BinFHELogicTest, TestNOT) {
  using namespace lbcrypto;

  // Test NOT operation
  std::vector<int> test_cases = {0, 1};

  for (int a : test_cases) {
    // Encrypt input
    auto ct_a =
        cryptoContext->Encrypt(secretKey, a, BINFHE_OUTPUT::SMALL_DIM, 8);

    // Compute NOT
    auto ct_result = not_bit(cryptoContext, ct_a);

    // Decrypt result
    LWEPlaintext result;
    cryptoContext->Decrypt(secretKey, ct_result, &result, 8);

    // Check result
    int expected = (!a) ? 1 : 0;
    EXPECT_EQ(result, expected) << "NOT(" << a << ") failed";
  }
}

TEST_F(BinFHELogicTest, TestFullAdderSum) {
  using namespace lbcrypto;

  // Test all full adder sum combinations
  for (int a = 0; a <= 1; a++) {
    for (int b = 0; b <= 1; b++) {
      for (int cin = 0; cin <= 1; cin++) {
        // Encrypt inputs
        auto ct_a =
            cryptoContext->Encrypt(secretKey, a, BINFHE_OUTPUT::SMALL_DIM, 8);
        auto ct_b =
            cryptoContext->Encrypt(secretKey, b, BINFHE_OUTPUT::SMALL_DIM, 8);
        auto ct_cin =
            cryptoContext->Encrypt(secretKey, cin, BINFHE_OUTPUT::SMALL_DIM, 8);

        // Compute full adder sum
        auto ct_sum = full_adder_sum(cryptoContext, ct_a, ct_b, ct_cin);

        // Decrypt result
        LWEPlaintext sum;
        cryptoContext->Decrypt(secretKey, ct_sum, &sum, 8);

        // Check result
        int expected_sum = (a + b + cin) & 1;

        EXPECT_EQ(sum, expected_sum) << "Full adder sum failed for a=" << a
                                     << ", b=" << b << ", cin=" << cin;
      }
    }
  }
}

TEST_F(BinFHELogicTest, TestFullAdderCarry) {
  using namespace lbcrypto;

  // Test all full adder carry combinations
  for (int a = 0; a <= 1; a++) {
    for (int b = 0; b <= 1; b++) {
      for (int cin = 0; cin <= 1; cin++) {
        // Encrypt inputs
        auto ct_a =
            cryptoContext->Encrypt(secretKey, a, BINFHE_OUTPUT::SMALL_DIM, 8);
        auto ct_b =
            cryptoContext->Encrypt(secretKey, b, BINFHE_OUTPUT::SMALL_DIM, 8);
        auto ct_cin =
            cryptoContext->Encrypt(secretKey, cin, BINFHE_OUTPUT::SMALL_DIM, 8);

        // Compute full adder carry
        auto ct_carry = full_adder_carry(cryptoContext, ct_a, ct_b, ct_cin);

        // Decrypt result
        LWEPlaintext carry;
        cryptoContext->Decrypt(secretKey, ct_carry, &carry, 8);

        // Check result
        int expected_carry = (a + b + cin) >> 1;

        EXPECT_EQ(carry, expected_carry)
            << "Full adder carry failed for a=" << a << ", b=" << b
            << ", cin=" << cin;
      }
    }
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
