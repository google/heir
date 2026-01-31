
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
std::vector<CiphertextT> loop(CryptoContextT cc, std::vector<CiphertextT> v0);
std::vector<CiphertextT> loop__encrypt__arg0(CryptoContextT cc, std::vector<float> v0, PublicKeyT pk);
std::vector<float> loop__decrypt__result0(CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk);
CryptoContextT loop__generate_crypto_context();
CryptoContextT loop__configure_crypto_context(CryptoContextT cc, PrivateKeyT sk);
