#include <vector>

#include "gtest/gtest.h"
#include "src/binfhe/include/binfhecontext.h"
#include "src/binfhe/include/lwe-ciphertext.h"
#include "src/binfhe/include/lwe-privatekey.h"
#include "src/binfhe/include/lwe-publickey.h"
#include "tests/Examples/openfhe/binfhe/or_bits_only_lib.h"

class BinFHELogicTest_OR : public ::testing::Test {
 protected:
  void SetUp() override {
    using namespace lbcrypto;
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, true, 12, 0, GINX, false);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cryptoContext = std::make_shared<BinFHEContext>(cc);
    secretKey = sk;
  }
  BinFHEContextT cryptoContext;
  lbcrypto::LWEPrivateKey secretKey;
};

TEST_F(BinFHELogicTest_OR, TestOR) {
  using namespace lbcrypto;
  std::vector<std::pair<int, int>> cases{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  for (auto [a, b] : cases) {
    auto ct_a =
        cryptoContext->Encrypt(secretKey, a, BINFHE_OUTPUT::SMALL_DIM, 8);
    auto ct_b =
        cryptoContext->Encrypt(secretKey, b, BINFHE_OUTPUT::SMALL_DIM, 8);
    auto ct = or_bits(cryptoContext, ct_a, ct_b);
    LWEPlaintext r;
    cryptoContext->Decrypt(secretKey, ct, &r, 8);
    EXPECT_EQ(r, (a | b) ? 1 : 0) << "OR(" << a << "," << b << ")";
  }
}
