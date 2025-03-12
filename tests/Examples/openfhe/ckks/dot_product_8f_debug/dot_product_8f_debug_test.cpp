#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/dot_product_8f_debug/dot_product_8f_debug_lib.h"

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

#define OP
#define DECRYPT
#define PRECISION

void __heir_debug(CryptoContextT cc, PrivateKeyT sk, CiphertextT ct,
                  const std::map<std::string, std::string>& debugAttrMap) {
#ifdef OP
  auto isBlockArgument = debugAttrMap.at("asm.is_block_arg");
  if (isBlockArgument == "1") {
    std::cout << "Input" << std::endl;
  } else {
    std::cout << debugAttrMap.at("asm.op_name") << std::endl;
  }
#endif

#ifdef DECRYPT
  PlaintextT ptxt;
  cc->Decrypt(sk, ct, &ptxt);
  ptxt->SetLength(std::stod(debugAttrMap.at("message.size")));
  std::vector<double> result;
  for (size_t i = 0; i < ptxt->GetLength(); i++) {
    result.push_back(ptxt->GetRealPackedValue()[i]);
  }

#ifdef PRECISION
  if (debugAttrMap.find("secret.execution_result") != debugAttrMap.end()) {
    auto plaintextResultStr = debugAttrMap.at("secret.execution_result");
    // plaintextResultStr has the form "[1.0, 2.0, 3.0]", parse it into vector
    // of double
    std::vector<double> plaintextResult;
    std::string value;
    for (size_t i = 1; i < plaintextResultStr.size() - 1; i++) {
      if (plaintextResultStr[i] == ',') {
        plaintextResult.push_back(std::stod(value));
        value.clear();
      } else {
        value += plaintextResultStr[i];
      }
    }
    plaintextResult.push_back(std::stod(value));

    double maxError = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < std::min(result.size(), plaintextResult.size());
         i++) {
      auto err = log2(std::abs(result[i] - plaintextResult[i]));
      maxError = std::max(maxError, err);
    }

    std::cout << "  Precision lost: 2^" << std::setprecision(3) << maxError
              << std::endl;
  }
#endif
#endif
}

namespace mlir {
namespace heir {
namespace openfhe {

TEST(DotProduct8FTest, RunTest) {
  auto cryptoContext = dot_product__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      dot_product__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<float> arg1 = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  float expected = 2.4 + 0.1;

  auto arg0Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto arg1Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg1, publicKey);
  auto outputEncrypted =
      dot_product(cryptoContext, secretKey, arg0Encrypted, arg1Encrypted);
  auto actual =
      dot_product__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_NEAR(expected, actual, 1e-3);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
