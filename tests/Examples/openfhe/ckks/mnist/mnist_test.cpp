#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"                               // from @googletest
#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/core/include/utils/inttypes.h"           // from @openfhe
#include "src/pke/include/key/keypair.h"               // from @openfhe
#include "src/pke/include/key/privatekey-fwd.h"        // from @openfhe

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/mnist/mnist_lib.h"

#define OP
#define DECRYPT

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
  result.reserve(ptxt->GetLength());
  for (size_t i = 0; i < ptxt->GetLength(); i++) {
    result.push_back(ptxt->GetRealPackedValue()[i]);
  }
  std::cout << "decrypted: [";
  for (auto val : result) {
    std::cout << std::setprecision(3) << (abs(val) < 1e-10 ? 0 : val) << ",";
  }
  std::cout << "]\n";
#endif
}

namespace mlir {
namespace heir {
namespace openfhe {

TEST(MNISTTest, RunTest) {
  auto cryptoContext = mnist__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = mnist__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0Vals(5, 1.0);  // input (0, 1, 2, 3, 4)
  for (int i = 0; i < 5; ++i) {
    arg0Vals[i] = i;
  }

  // Set matrix main diagonal = 1
  std::vector<float> matVals(3 * 5, 0.0);
  matVals[0] = 1.0;
  matVals[1 + 5 * 1] = 1.0;
  matVals[2 + 5 * 2] = 1.0;

  // The first elements of the vector.
  std::vector<float> expected = {0, 1, 2};

  auto arg0Encrypted = mnist__encrypt__arg1(cryptoContext, arg0Vals, publicKey);

  // Insert timing info
  std::clock_t cStart = std::clock();
  auto outputEncrypted =
      mnist(cryptoContext, secretKey, matVals, arg0Encrypted);
  std::clock_t cEnd = std::clock();
  double timeElapsedMs = 1000.0 * (cEnd - cStart) / CLOCKS_PER_SEC;
  std::cout << "CPU time used: " << timeElapsedMs << " ms\n";

  auto actual =
      mnist__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[i], 1e-6);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
