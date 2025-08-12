#include "lib/Parameters/RLWEParams.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <numeric>
#include <sstream>
#include <vector>

#include "lib/Parameters/RLWESecurityParams.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "src/core/include/openfhecore.h"           // from @openfhe

namespace mlir {
namespace heir {

// from OpenFHE, the empirical way to select dnum based on level
int computeDnum(int level) {
  if (level > 3) {
    return 3;
  }
  if (level > 0) {
    return 2;
  }
  return 1;
}

RLWESchemeParam RLWESchemeParam::getConservativeRLWESchemeParam(
    int level, int minRingDim, bool usePublicKey,
    bool encryptionTechniqueExtended) {
  auto logModuli = 60;  // assume all 60 bit moduli
  auto dnum = computeDnum(level);
  std::vector<double> logqi(level + 1, logModuli);
  std::vector<double> logpi(ceil(static_cast<double>(logqi.size()) / dnum),
                            logModuli);

  auto totalQP = logModuli * (logqi.size() + logpi.size());

  auto ringDim = computeRingDim(totalQP, minRingDim);

  return RLWESchemeParam(ringDim, level, logqi, dnum, logpi, usePublicKey,
                         encryptionTechniqueExtended);
}

int64_t findPrime(int qi, int ringDim,
                  const std::vector<int64_t>& existingPrimes) {
  while (qi < 80) {
    try {
      // openfhe FirstPrime will throw exception if it fails to find a prime
      bool redo = false;
      int64_t dupPrime;
      do {
        int64_t prime;
        if (!redo) {
          // first time, use first prime
          auto res =
              lbcrypto::FirstPrime<lbcrypto::NativeInteger>(qi, 2 * ringDim);
          prime = res.ConvertToInt();
        } else {
          // start from the duplicated prime
          auto res = lbcrypto::NextPrime(lbcrypto::NativeInteger(dupPrime),
                                         2 * ringDim);
          prime = res.ConvertToInt();
        }
        if (std::find(existingPrimes.begin(), existingPrimes.end(), prime) ==
            existingPrimes.end()) {
          return prime;
        }
        dupPrime = prime;
        redo = true;
      } while (redo);
    } catch (...) {
      qi += 1;
    }
  }
  assert(false && "failed to generate good qi");
  return 0;
}

RLWESchemeParam RLWESchemeParam::getConcreteRLWESchemeParam(
    std::vector<double> logqi, int minRingDim, bool usePublicKey,
    bool encryptionTechniqueExtended, int64_t plaintextModulus) {
  auto level = logqi.size() - 1;
  auto dnum = computeDnum(level);

  // sanitize qi
  for (auto& qi : logqi) {
    if (qi < 20) {
      qi = 20;
    }
  }

  auto maxLogqi = *std::max_element(logqi.begin(), logqi.end());
  // make P > Q / dnum
  std::vector<double> logpi(ceil(static_cast<double>(logqi.size()) / dnum),
                            maxLogqi);

  double logPQ = std::accumulate(logqi.begin(), logqi.end(), 0.0) +
                 std::accumulate(logpi.begin(), logpi.end(), 0.0);

  // ringDim will change if newLogPQ is too large
  auto ringDim = computeRingDim(logPQ, minRingDim);
  std::vector<int64_t> qiImpl;
  std::vector<int64_t> piImpl;
  bool redo = false;
  do {
    redo = false;
    qiImpl.clear();
    piImpl.clear();

    std::vector<int64_t> existingPrimes;
    // special treatment for BGV plaintext modulus
    if (plaintextModulus != 0) {
      existingPrimes.push_back(plaintextModulus);
    }

    double newLogPQ = 0;
    for (auto qi : logqi) {
      auto prime = findPrime(qi, ringDim, existingPrimes);
      qiImpl.push_back(prime);
      existingPrimes.push_back(prime);
      newLogPQ += log2(prime);
    }
    for (auto pi : logpi) {
      auto prime = findPrime(pi, ringDim, existingPrimes);
      piImpl.push_back(prime);
      existingPrimes.push_back(prime);
      newLogPQ += log2(prime);
    }
    // if generated primes are too large, increase ringDim
    auto newRingDim = computeRingDim(newLogPQ, minRingDim);
    if (newRingDim != ringDim) {
      ringDim = newRingDim;
      redo = true;
    }
  } while (redo);

  // update logqi and logpi
  logqi.clear();
  logpi.clear();
  for (auto qi : qiImpl) {
    logqi.push_back(log2(qi));
  }
  for (auto pi : piImpl) {
    logpi.push_back(log2(pi));
  }

  return RLWESchemeParam(ringDim, level, logqi, qiImpl, dnum, logpi, piImpl,
                         usePublicKey, encryptionTechniqueExtended);
}

void RLWESchemeParam::print(llvm::raw_ostream& os) const {
  os << "ringDim: " << ringDim << "\n";
  os << "level: " << level << "\n";
  os << "logqi: ";
  for (auto qi : logqi) {
    os << doubleToString2Prec(qi) << " ";
  }
  os << "\n";
  os << "qi: ";
  for (auto qi : qi) {
    os << qi << " ";
  }
  os << "\n";
  os << "dnum: " << dnum << "\n";
  os << "logpi: ";
  for (auto pi : logpi) {
    os << doubleToString2Prec(pi) << " ";
  }
  os << "\n";
  os << "pi: ";
  for (auto pi : pi) {
    os << pi << " ";
  }
  os << "\n";
  os << "usePublicKey: " << usePublicKey << "\n";
  os << "encryptionTechniqueExtended: " << encryptionTechniqueExtended << "\n";
}

}  // namespace heir
}  // namespace mlir
