#ifndef THIRD_PARTY_HEIR_TESTS_EXAMPLES_OPENFHE_CKKS_DOT_PRODUCT_8F_DEBUG_DEBUG_HELPER_H_
#define THIRD_PARTY_HEIR_TESTS_EXAMPLES_OPENFHE_CKKS_DOT_PRODUCT_8F_DEBUG_DEBUG_HELPER_H_

#include <map>
#include <string>

#include "src/pke/include/openfhe.h"  // from @openfhe

using CryptoContextT = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using CiphertextT = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using PrivateKeyT = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;

void __heir_debug(CryptoContextT cc, PrivateKeyT sk, CiphertextT ct,
                  const std::map<std::string, std::string>& debugAttrMap);

#endif  // THIRD_PARTY_HEIR_TESTS_EXAMPLES_OPENFHE_CKKS_DOT_PRODUCT_8F_DEBUG_DEBUG_HELPER_H_
