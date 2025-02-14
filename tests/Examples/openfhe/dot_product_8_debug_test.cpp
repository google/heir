#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <vector>

#include "gtest/gtest.h"                               // from @googletest
#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/core/include/utils/inttypes.h"           // from @openfhe
#include "src/pke/include/key/privatekey-fwd.h"        // from @openfhe
#include "src/pke/include/openfhe.h"                   // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/dot_product_8_debug_lib.h"

// DecryptCore not accessible from CryptoContext
// so copy from @openfhe//src/pke/lib/schemerns/rns-pke.cpp
DCRTPoly DecryptCore(const std::vector<DCRTPoly>& cv,
                     const PrivateKey<DCRTPoly> privateKey) {
  const DCRTPoly& s = privateKey->GetPrivateElement();

  size_t sizeQ = s.GetParams()->GetParams().size();
  size_t sizeQl = cv[0].GetParams()->GetParams().size();

  size_t diffQl = sizeQ - sizeQl;

  auto scopy(s);
  scopy.DropLastElements(diffQl);

  DCRTPoly sPower(scopy);

  DCRTPoly b(cv[0]);
  b.SetFormat(Format::EVALUATION);

  DCRTPoly ci;
  for (size_t i = 1; i < cv.size(); i++) {
    ci = cv[i];
    ci.SetFormat(Format::EVALUATION);

    b += sPower * ci;
    sPower *= scopy;
  }
  return b;
}

#define DECRYPT
#define NOISE

void __heir_debug(CryptoContextT cc, PrivateKeyT sk, CiphertextT ct) {
#ifdef DECRYPT
  PlaintextT ptxt;
  cc->Decrypt(sk, ct, &ptxt);
  ptxt->SetLength(8);
  std::cout << ptxt << std::endl;
#endif

#ifdef NOISE
  auto cv = ct->GetElements();
  size_t sizeQl = cv[0].GetParams()->GetParams().size();

  auto b = DecryptCore(cv, sk);
  b.SetFormat(Format::COEFFICIENT);

  double noise = (log2(b.Norm()));

  double logQ = 0;
  std::vector<double> logqi_v;
  for (usint i = 0; i < sizeQl; i++) {
    double logqi =
        log2(cv[0].GetParams()->GetParams()[i]->GetModulus().ConvertToInt());
    logqi_v.push_back(logqi);
    logQ += logqi;
  }

  std::cout << "cv " << cv.size() << " Ql " << sizeQl << " logQ: " << logQ
            << " logqi: " << logqi_v << " budget " << logQ - noise - 1
            << " noise: " << noise << std::endl;
#endif
}

namespace mlir {
namespace heir {
namespace openfhe {

TEST(DotProduct8Test, RunTest) {
  auto cryptoContext = dot_product__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      dot_product__configure_crypto_context(cryptoContext, secretKey);

  std::vector<int16_t> arg0 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int16_t> arg1 = {2, 3, 4, 5, 6, 7, 8, 9};
  int64_t expected = 240;

  auto arg0Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto arg1Encrypted =
      dot_product__encrypt__arg1(cryptoContext, arg1, publicKey);
  auto outputEncrypted =
      dot_product(cryptoContext, secretKey, arg0Encrypted, arg1Encrypted);
  auto actual =
      dot_product__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_EQ(expected, actual);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
