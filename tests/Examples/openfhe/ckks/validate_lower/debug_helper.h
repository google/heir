#ifndef TESTS_EXAMPLES_OPENFHE_CKKS_VALIDATE_LOWER_DEBUG_HELPER_H_
#define TESTS_EXAMPLES_OPENFHE_CKKS_VALIDATE_LOWER_DEBUG_HELPER_H_

#include <map>
#include <string>
#include <vector>

#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/pke/include/ciphertext-fwd.h"            // from @openfhe
#include "src/pke/include/cryptocontext-fwd.h"         // from @openfhe
#include "src/pke/include/key/privatekey-fwd.h"        // from @openfhe

using CiphertextT = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using CryptoContextT = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using PrivateKeyT = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;

void __heir_debug(CryptoContextT cc, PrivateKeyT sk, CiphertextT ct,
                  const std::map<std::string, std::string>& debugAttrMap);

void __heir_debug(CryptoContextT cc, PrivateKeyT sk,
                  std::vector<CiphertextT> ct,
                  const std::map<std::string, std::string>& debugAttrMap);

#endif  // TESTS_EXAMPLES_OPENFHE_CKKS_VALIDATE_LOWER_DEBUG_HELPER_H_
