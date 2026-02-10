#include "tests/Examples/openfhe/ckks/loop_support/debug_helper.h"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/pke/include/encoding/plaintext-fwd.h"    // from @openfhe

using lbcrypto::DCRTPoly;
using PlaintextT = lbcrypto::Plaintext;

void __heir_debug(CryptoContextT cc, PrivateKeyT sk, CiphertextT ct,
                  const std::map<std::string, std::string>& debugAttrMap) {
  auto isBlockArgument = debugAttrMap.at("asm.is_block_arg");
  if (isBlockArgument == "1") {
    std::cout << "Input" << std::endl;
  } else {
    std::cout << debugAttrMap.at("asm.result_ssa_format") << std::endl;
  }

  PlaintextT ptxt;
  cc->Decrypt(sk, ct, &ptxt);
  ptxt->SetLength(std::stod(debugAttrMap.at("message.size")));
  std::vector<double> result;
  result.reserve(ptxt->GetLength());
  for (size_t i = 0; i < ptxt->GetLength(); i++) {
    result.push_back(ptxt->GetRealPackedValue()[i]);
  }

  std::cout << "  Decrypted: [";
  for (double val : result) {
    std::cout << std::setprecision(3) << val << ", ";
  }
  std::cout << "]\n";
  std::cout << "  Scale: " << log2(ct->GetScalingFactor()) << std::endl;
}
