// This file and the corresponding cpp file were copied from the compiled
// output of the e2e test in tests/Examples/openfhe/ckks/loop_support

#include "src/pke/include/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = Ciphertext<DCRTPoly>;
using ConstCiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

std::vector<float> _assign_layout_6191183397986546506(std::vector<float> v0);

// A simple test for loop support.
//
// Computes the function
//
//     def f(x):
//       sum = 1.0
//       for i in range(8):
//         sum = sum * x - 1.0
//       return sum
//
std::vector<CiphertextT> loop(CryptoContextT cc, std::vector<CiphertextT> v0);

std::vector<CiphertextT> loop__encrypt__arg0(CryptoContextT cc,
                                             std::vector<float> v0,
                                             PublicKeyT pk);
std::vector<float> loop__decrypt__result0(CryptoContextT cc,
                                          std::vector<CiphertextT> v0,
                                          PrivateKeyT sk);
CryptoContextT loop__generate_crypto_context();
CryptoContextT loop__configure_crypto_context(CryptoContextT cc,
                                              PrivateKeyT sk);
