#include <vector>

#include "gtest/gtest.h"
#include "src/binfhe/include/binfhecontext.h"
#include "src/binfhe/include/lwe-ciphertext.h"
#include "src/binfhe/include/lwe-privatekey.h"
#include "src/binfhe/include/lwe-publickey.h"
#include "tests/Examples/openfhe/binfhe/not_bit_only_lib.h"

class BinFHELogicTest_NOT : public ::testing::Test {
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

TEST_F(BinFHELogicTest_NOT, TestNOT) {
  using namespace lbcrypto;
  for (int a : {0, 1}) {
    auto ct_a =
        cryptoContext->Encrypt(secretKey, a, BINFHE_OUTPUT::SMALL_DIM, 8);
    auto ct = not_bit(cryptoContext, ct_a);
    LWEPlaintext r;
    cryptoContext->Decrypt(secretKey, ct, &r, 8);
    EXPECT_EQ(r, (!a) ? 1 : 0) << "NOT(" << a << ")";
  }
}
