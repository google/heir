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
#include "src/core/include/utils/inttypes.h"           // from @openfhe
#include "src/pke/include/encoding/plaintext-fwd.h"    // from @openfhe

using lbcrypto::DCRTPoly;
using PlaintextT = lbcrypto::Plaintext;

// DecryptCore not accessible from CryptoContext
// so copy from @openfhe//src/pke/lib/schemerns/rns-pke.cpp
DCRTPoly DecryptCore(const std::vector<DCRTPoly>& cv,
                     const PrivateKeyT privateKey) {
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

void __heir_debug(CryptoContextT cc, PrivateKeyT sk, CiphertextT ct,
                  const std::map<std::string, std::string>& debugAttrMap) {
#ifdef OP
  auto isBlockArgument = debugAttrMap.at("asm.is_block_arg");
  if (isBlockArgument == "1") {
    std::cout << "Input" << std::endl;
  } else {
    std::cout << debugAttrMap.at("asm.result_ssa_format") << std::endl;
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

  std::cout << "  Decrypted: [";
  for (double val : result) {
    std::cout << std::setprecision(3) << val << ", ";
  }
  std::cout << "]\n";

  // print the scale
  std::cout << "  Scale: " << log2(ct->GetScalingFactor()) << std::endl;
#endif
}

