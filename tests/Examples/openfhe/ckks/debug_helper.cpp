#include "tests/Examples/openfhe/ckks/debug_helper.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
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

  // print the scale
  std::cout << "  Scale: " << log2(ct->GetScalingFactor()) << std::endl;

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

    // full packed pt
    // Note that packing behavior is different in OpenfhePkeEmitter
    // and plaintext backend, care should be taken...
    std::vector<double> packed;
    for (size_t i = 0;
         i <
         cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
         i++) {
      packed.push_back(plaintextResult[i % plaintextResult.size()]);
    }
    auto packedPt = cc->MakeCKKSPackedPlaintext(packed);
    // EvalSub itself will adjust the scale of packedPt
    auto zeroCt = cc->EvalSub(ct, packedPt);
    auto b = DecryptCore(zeroCt->GetElements(), sk);
    b.SetFormat(Format::COEFFICIENT);
    double noise = (log2(b.Norm()));
    std::cout << "  Noise: 2^" << std::setprecision(3) << noise << std::endl;
  }
#endif
#endif
}
