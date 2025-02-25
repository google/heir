#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"                                 // from @googletest
#include "src/core/include/lattice/hal/lat-backend.h"    // from @openfhe
#include "src/core/include/math/hal/nativeintbackend.h"  // from @openfhe
#include "src/core/include/utils/inttypes.h"             // from @openfhe
#include "src/pke/include/encoding/plaintext-fwd.h"      // from @openfhe
#include "src/pke/include/key/privatekey-fwd.h"          // from @openfhe
#include "src/pke/include/scheme/bfvrns/bfvrns-cryptoparameters.h"  // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/bfv/dot_product_8_debug/dot_product_8_debug_lib.h"

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
#define NOISE

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
  std::cout << "  " << ptxt << std::endl;
#endif

#ifdef NOISE
  auto cv = ct->GetElements();
  size_t sizeQl = cv[0].GetParams()->GetParams().size();

  auto b = DecryptCore(cv, sk);
  b.SetFormat(Format::COEFFICIENT);

  // B/FV specific
  // from @openfhe//src/pke/extras/bfv-mult-bug.cpp
  const auto cryptoParams = std::static_pointer_cast<CryptoParametersBFVRNS>(
      sk->GetCryptoParameters());

  const auto encParams = cryptoParams->GetElementParams();
  NativeInteger NegQModt = cryptoParams->GetNegQModt();
  NativeInteger NegQModtPrecon = cryptoParams->GetNegQModtPrecon();
  const NativeInteger t = cryptoParams->GetPlaintextModulus();
  std::vector<NativeInteger> tInvModq = cryptoParams->GettInvModq();

  // Get a new plaintext with full slots
  // SetLength(8) above will truncate the plaintext
  PlaintextT newPtxt;
  cc->Decrypt(sk, ct, &newPtxt);
  // Repack to convert from NativePoly to DCRTPoly
  std::vector<int64_t> value = newPtxt->GetPackedValue();
  Plaintext repack = cc->MakePackedPlaintext(value);
  DCRTPoly plain = repack->GetElement<DCRTPoly>();
  plain.SetFormat(Format::COEFFICIENT);
  plain.TimesQovert(encParams, tInvModq, t, NegQModt, NegQModtPrecon);

  // remove the message, leave only the noise
  DCRTPoly res;
  res = b - plain;

  double noise = (log2(res.Norm()));

  double logQ = 0;
  std::vector<double> logqi_v;
  for (usint i = 0; i < sizeQl; i++) {
    double logqi =
        log2(cv[0].GetParams()->GetParams()[i]->GetModulus().ConvertToInt());
    logqi_v.push_back(logqi);
    logQ += logqi;
  }

  auto logT = log2(t.ConvertToInt());

  std::cout << "  cv " << cv.size() << " Ql " << sizeQl
            << " log(Q/2T): " << logQ - logT - 1 << " logqi: " << logqi_v
            << " budget " << logQ - logT - 1 - noise << " noise: " << noise
            << std::endl;

  // print the predicted bound by analysis
  if (debugAttrMap.find("noise.bound") != debugAttrMap.end()) {
    double noiseBound = std::stod(debugAttrMap.at("noise.bound"));

    std::cout << "  noise bound: " << noiseBound
              << "  gap: " << noiseBound - noise << std::endl;
  }
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
