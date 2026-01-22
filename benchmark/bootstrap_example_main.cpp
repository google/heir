#include <iostream>
#include <vector>

#include "benchmark/bootstrap_example.h"

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
int main(int argc, char** argv) {
  std::cout << "Bootstrap Example Main\n";
  std::cout << "Generating crypto context\n";
  auto cryptoContext = loop__generate_crypto_context();
  std::cout << "Generating keys\n";
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  std::cout << "Configuring crypto context\n";
  cryptoContext = loop__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0 = {0.,         0.14285714, 0.28571429, 0.42857143,
                             0.57142857, 0.71428571, 0.85714286, 1.};
  std::vector<float> expected = {-1.,         -1.16666629, -1.39989342,
                                 -1.74687019, -2.29543899, -3.19507837,
                                 -4.66914279, -7.};

  std::cout << "Encrypting arg\n";
  auto arg0Encrypted = loop__encrypt__arg0(cryptoContext, arg0, publicKey);
  std::cout << "Running loop\n";
  auto outputEncrypted = loop(cryptoContext, arg0Encrypted);
  std::cout << "Decrypting result\n";
  auto actual =
      loop__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  for (size_t i = 0; i < actual.size(); i++) {
    std::cout << "actual[" << i << "] = " << actual[i] << ", expected[" << i
              << "] = " << expected[i] << "\n";
  }

  return 0;
}
