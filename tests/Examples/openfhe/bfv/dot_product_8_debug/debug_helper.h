#ifndef TESTS_EXAMPLES_OPENFHE_BFV_DOT_PRODUCT_8_DEBUG_DEBUG_HELPER_H_
#define TESTS_EXAMPLES_OPENFHE_BFV_DOT_PRODUCT_8_DEBUG_DEBUG_HELPER_H_

#include <map>
#include <string>

#include "src/pke/include/openfhe.h"  // from @openfhe

using CiphertextT = lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>;
using CryptoContextT = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using PrivateKeyT = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;

void __heir_debug(CryptoContextT cc, PrivateKeyT sk, CiphertextT ct,
                  const std::map<std::string, std::string>& debugAttrMap);

#endif  // TESTS_EXAMPLES_OPENFHE_BFV_DOT_PRODUCT_8_DEBUG_DEBUG_HELPER_H_
