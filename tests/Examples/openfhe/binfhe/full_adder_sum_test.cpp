#include <vector>

#include "gtest/gtest.h"
#include "src/binfhe/include/binfhecontext.h"
#include "src/binfhe/include/lwe-ciphertext.h"
#include "src/binfhe/include/lwe-privatekey.h"
#include "src/binfhe/include/lwe-publickey.h"
#include "tests/Examples/openfhe/binfhe/full_adder_sum_lib.h"

class BinFHELogicTest_FullAdderSum : public ::testing::Test {
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

TEST_F(BinFHELogicTest_FullAdderSum, TestFullAdderSum) {
  using namespace lbcrypto;
  for (int a = 0; a <= 1; a++)
    for (int b = 0; b <= 1; b++)
      for (int cin = 0; cin <= 1; cin++) {
        auto ct_a =
            cryptoContext->Encrypt(secretKey, a, BINFHE_OUTPUT::SMALL_DIM, 8);
        auto ct_b =
            cryptoContext->Encrypt(secretKey, b, BINFHE_OUTPUT::SMALL_DIM, 8);
        auto ct_cin =
            cryptoContext->Encrypt(secretKey, cin, BINFHE_OUTPUT::SMALL_DIM, 8);
        auto ct = full_adder_sum(cryptoContext, ct_a, ct_b, ct_cin);
        LWEPlaintext r;
        cryptoContext->Decrypt(secretKey, ct, &r, 8);
        int expected = (a + b + cin) & 1;
        EXPECT_EQ(r, expected) << "sum(" << a << "," << b << "," << cin << ")";
      }
}
