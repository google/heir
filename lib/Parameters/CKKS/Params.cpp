#include "lib/Parameters/CKKS/Params.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Parameters/RLWEParams.h"
#include "lib/Parameters/RLWESecurityParams.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "src/core/include/openfhecore.h"           // from @openfhe

namespace mlir {
namespace heir {
namespace ckks {

/// By original we mean the method in RNS-CKKS implementation
/// Corresponds to FIXED* in OpenFHE
static std::vector<int64_t> moduliQGenerationOpenFHEFixed(int logFirstMod,
                                                          int logDefaultScale,
                                                          int numLevel,
                                                          int ringDim) {
  auto cyclOrder = ringDim * 2;
  std::vector<int64_t> moduliQ(numLevel);
  lbcrypto::NativeInteger q =
      lbcrypto::FirstPrime<NativeInteger>(logDefaultScale, cyclOrder);
  moduliQ[numLevel - 1] = q.ConvertToInt();

  auto maxPrime{q};
  auto minPrime{q};

  auto qPrev = q;
  auto qNext = q;
  if (numLevel > 2) {
    for (size_t i = numLevel - 2, cnt = 0; i >= 1; --i, ++cnt) {
      if ((cnt % 2) == 0) {
        qPrev = PreviousPrime(qPrev, cyclOrder);
        moduliQ[i] = qPrev.ConvertToInt();
      } else {
        qNext = NextPrime(qNext, cyclOrder);
        moduliQ[i] = qNext.ConvertToInt();
      }

      if (moduliQ[i] > maxPrime)
        maxPrime = moduliQ[i];
      else if (moduliQ[i] < minPrime)
        minPrime = moduliQ[i];
    }
  }

  if (logFirstMod == logDefaultScale) {  // this requires dcrtBits < 60
    moduliQ[0] =
        lbcrypto::NextPrime<NativeInteger>(maxPrime, cyclOrder).ConvertToInt();
  } else {
    moduliQ[0] = lbcrypto::LastPrime<NativeInteger>(logFirstMod, cyclOrder)
                     .ConvertToInt();

    // find if the value of moduliQ[0] is already in the vector starting with
    // moduliQ[1] and if there is, then get another prime for moduliQ[0]
    const auto pos = std::find(moduliQ.begin() + 1, moduliQ.end(), moduliQ[0]);
    if (pos != moduliQ.end()) {
      moduliQ[0] = lbcrypto::NextPrime<NativeInteger>(maxPrime, cyclOrder)
                       .ConvertToInt();
    }
  }
  return moduliQ;
}

/// See "Reduced Error" paper https://eprint.iacr.org/2020/1118
/// Corresponds to FLEXIBLE* in OpenFHE
static std::vector<int64_t> moduliQGenerationReducedError(int logFirstMod,
                                                          int logDefaultScale,
                                                          int numLevel,
                                                          int ringDim) {
  auto cyclOrder = ringDim * 2;
  std::vector<int64_t> moduliQ(numLevel);
  lbcrypto::NativeInteger q =
      lbcrypto::FirstPrime<lbcrypto::NativeInteger>(logDefaultScale, cyclOrder);
  moduliQ[numLevel - 1] = q.ConvertToInt();

  auto maxPrime{q};
  auto minPrime{q};

  if (numLevel > 2) {
    for (size_t i = numLevel - 2, cnt = 0; i >= 1; --i, ++cnt) {
      // Comments from OpenFHE ckksrns-parametergeneration.cpp
      /* Scaling factors in FLEXIBLEAUTO are a bit fragile,
       * in the sense that once one scaling factor gets far enough from the
       * original scaling factor, subsequent level scaling factors quickly
       * diverge to either 0 or infinity. To mitigate this problem to a certain
       * extend, we have a special prime selection process in place. The goal is
       * to maintain the scaling factor of all levels as close to the original
       * scale factor of level 0 as possible.
       */
      double sf = static_cast<double>(moduliQ[numLevel - 1]);
      for (size_t i = numLevel - 2, cnt = 0; i >= 1; --i, ++cnt) {
        sf = pow(sf, 2) / static_cast<double>(moduliQ[i + 1]);
        NativeInteger sfInt = std::llround(sf);
        NativeInteger sfRem = sfInt.Mod(cyclOrder);
        bool hasSameMod = true;
        if ((cnt % 2) == 0) {
          NativeInteger qPrev =
              sfInt - NativeInteger(cyclOrder) - sfRem + NativeInteger(1);
          while (hasSameMod) {
            hasSameMod = false;
            qPrev = lbcrypto::PreviousPrime(qPrev, cyclOrder);
            for (size_t j = i + 1; j < numLevel; j++) {
              if (qPrev == moduliQ[j]) {
                hasSameMod = true;
                break;
              }
            }
          }
          moduliQ[i] = qPrev.ConvertToInt();
        } else {
          NativeInteger qNext =
              sfInt + NativeInteger(cyclOrder) - sfRem + NativeInteger(1);
          while (hasSameMod) {
            hasSameMod = false;
            qNext = lbcrypto::NextPrime(qNext, cyclOrder);
            for (size_t j = i + 1; j < numLevel; j++) {
              if (qNext == moduliQ[j]) {
                hasSameMod = true;
                break;
              }
            }
          }
          moduliQ[i] = qNext.ConvertToInt();
        }
        if (moduliQ[i] > maxPrime)
          maxPrime = moduliQ[i];
        else if (moduliQ[i] < minPrime)
          minPrime = moduliQ[i];
      }
    }
  }

  if (logFirstMod == logDefaultScale) {  // this requires dcrtBits < 60
    moduliQ[0] =
        lbcrypto::NextPrime<lbcrypto::NativeInteger>(maxPrime, cyclOrder)
            .ConvertToInt();
  } else {
    moduliQ[0] =
        lbcrypto::LastPrime<lbcrypto::NativeInteger>(logFirstMod, cyclOrder)
            .ConvertToInt();

    // find if the value of moduliQ[0] is already in the vector starting with
    // moduliQ[1] and if there is, then get another prime for moduliQ[0]
    const auto pos = std::find(moduliQ.begin() + 1, moduliQ.end(), moduliQ[0]);
    if (pos != moduliQ.end()) {
      moduliQ[0] =
          lbcrypto::NextPrime<lbcrypto::NativeInteger>(maxPrime, cyclOrder)
              .ConvertToInt();
    }
  }
  return moduliQ;
}

// numScaleMod is L
SchemeParam SchemeParam::getConcreteSchemeParam(
    int logFirstMod, int logDefaultScale, int numScaleMod, int slotNumber,
    bool usePublicKey, bool encryptionTechniqueExtended, bool reducedError) {
  // CKKS slot number = ringDim / 2
  auto minRingDim = 2 * slotNumber;

  auto dnum = computeDnum(numScaleMod);

  // q0 + (q1 + ... + qL) = firstModBits + scalingModBits * L
  double logQ = logFirstMod + logDefaultScale * numScaleMod;
  // pi can be large
  auto sizePi = 60;

  // make P > Q / dnum
  auto logP = ceil(logQ / dnum);
  auto numPi = ceil(logP / sizePi);
  // update logP
  logP = numPi * sizePi;

  auto logPQ = logQ + logP;

  // ringDim will change if newLogPQ is too large
  auto ringDim = computeRingDim(logPQ, minRingDim);
  bool redo = false;
  std::vector<int64_t> qiImpl;
  std::vector<int64_t> piImpl;
  do {
    redo = false;
    qiImpl.clear();
    piImpl.clear();

    if (reducedError) {
      qiImpl = moduliQGenerationReducedError(logFirstMod, logDefaultScale,
                                             numScaleMod + 1, ringDim);
    } else {
      qiImpl = moduliQGenerationOpenFHEFixed(logFirstMod, logDefaultScale,
                                             numScaleMod + 1, ringDim);
    }
    std::vector<int64_t> existingPrimes = qiImpl;
    for (size_t i = 0; i < numPi; ++i) {
      auto prime = findPrime(sizePi, ringDim, existingPrimes);
      piImpl.push_back(prime);
      existingPrimes.push_back(prime);
    }

    // if generated primes are too large, increase ringDim
    double newLogPQ = 0;
    for (auto qi : qiImpl) {
      newLogPQ += log2(qi);
    }
    for (auto pi : piImpl) {
      newLogPQ += log2(pi);
    }
    auto newRingDim = computeRingDim(newLogPQ, minRingDim);
    if (newRingDim != ringDim) {
      ringDim = newRingDim;
      redo = true;
    }
  } while (redo);

  std::vector<double> logqi;
  std::vector<double> logpi;
  for (auto qi : qiImpl) {
    logqi.push_back(log2(qi));
  }
  for (auto pi : piImpl) {
    logpi.push_back(log2(pi));
  }

  return SchemeParam(
      RLWESchemeParam(ringDim, numScaleMod + 1, logqi, qiImpl, dnum, logpi,
                      piImpl, usePublicKey, encryptionTechniqueExtended),
      logDefaultScale);
}

SchemeParam SchemeParam::getSchemeParamFromAttr(SchemeParamAttr attr) {
  auto logN = attr.getLogN();
  auto ringDim = pow(2, logN);
  auto Q = attr.getQ();
  auto P = attr.getP();
  auto logDefaultScale = attr.getLogDefaultScale();
  std::vector<int64_t> qiImpl;
  std::vector<int64_t> piImpl;
  std::vector<double> logqi;
  std::vector<double> logpi;
  for (auto qi : Q.asArrayRef()) {
    qiImpl.push_back(qi);
    logqi.push_back(log2(qi));
  }
  for (auto pi : P.asArrayRef()) {
    piImpl.push_back(pi);
    logpi.push_back(log2(pi));
  }
  auto level = logqi.size() - 1;
  auto dnum = ceil(static_cast<double>(qiImpl.size()) / piImpl.size());
  auto usePublicKey = attr.getEncryptionType() == CKKSEncryptionType::pk;
  auto encryptionTechniqueExtended =
      attr.getEncryptionTechnique() == CKKSEncryptionTechnique::extended;
  return SchemeParam(
      RLWESchemeParam(ringDim, level, logqi, qiImpl, dnum, logpi, piImpl,
                      usePublicKey, encryptionTechniqueExtended),
      logDefaultScale);
}

void SchemeParam::print(llvm::raw_ostream& os) const {
  os << "logDefaultScale: " << logDefaultScale << "\n";
  RLWESchemeParam::print(os);
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
