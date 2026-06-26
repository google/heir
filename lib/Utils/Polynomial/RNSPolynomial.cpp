#include "lib/Utils/Polynomial/RNSPolynomial.h"

#include <cassert>
#include <utility>
#include <vector>

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/RNS/IR/RNSAttributes.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Polynomial/NTT.h"
#include "llvm/include/llvm/ADT/APInt.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

RNSPolynomial::RNSPolynomial(llvm::SmallVector<uint64_t> data,
                             llvm::SmallVector<uint64_t> moduli,
                             Form representation)
    : data(std::move(data)),
      moduli(std::move(moduli)),
      representation(representation) {
  assert(!this->moduli.empty() && "moduli cannot be empty");
  assert(this->data.size() % this->moduli.size() == 0 &&
         "numLimbs must divide data size");
  numCoeffs = this->data.size() / this->moduli.size();
}

RNSPolynomial RNSPolynomial::add(const RNSPolynomial& other) const {
  assert(representation == other.representation &&
         "Representations must match for arithmetic");
  assert(moduli == other.moduli && "Moduli must match for addition");
  assert(numCoeffs == other.numCoeffs && "Number of coefficients must match");

  llvm::SmallVector<uint64_t> resultData;
  resultData.reserve(data.size());

  for (size_t limbIdx = 0; limbIdx < getNumLimbs(); ++limbIdx) {
    uint64_t modulus = moduli[limbIdx];
    for (size_t coeffIdx = 0; coeffIdx < numCoeffs; ++coeffIdx) {
      uint64_t a = getElement(limbIdx, coeffIdx);
      uint64_t b = other.getElement(limbIdx, coeffIdx);
      uint64_t sum = a + b;
      if (sum >= modulus) {
        sum -= modulus;
      }
      resultData.push_back(sum);
    }
  }

  return RNSPolynomial(std::move(resultData), moduli, representation);
}

RNSPolynomial RNSPolynomial::sub(const RNSPolynomial& other) const {
  assert(representation == other.representation &&
         "Representations must match for arithmetic");
  assert(moduli == other.moduli && "Moduli must match for subtraction");
  assert(numCoeffs == other.numCoeffs && "Number of coefficients must match");

  llvm::SmallVector<uint64_t> resultData;
  resultData.reserve(data.size());

  for (size_t limbIdx = 0; limbIdx < getNumLimbs(); ++limbIdx) {
    uint64_t modulus = moduli[limbIdx];
    for (size_t coeffIdx = 0; coeffIdx < numCoeffs; ++coeffIdx) {
      uint64_t a = getElement(limbIdx, coeffIdx);
      uint64_t b = other.getElement(limbIdx, coeffIdx);
      uint64_t diff = a;
      if (diff < b) {
        diff += modulus;
      }
      diff -= b;
      resultData.push_back(diff);
    }
  }

  return RNSPolynomial(std::move(resultData), moduli, representation);
}

RNSPolynomial RNSPolynomial::mul(const RNSPolynomial& other) const {
  assert(moduli == other.moduli && "Moduli must match for multiplication");
  assert(numCoeffs == other.numCoeffs && "Number of coefficients must match");
  assert(representation == other.representation &&
         "Mismatched representations or unsupported conversion");

  if (representation == Form::EVAL && other.representation == Form::EVAL) {
    llvm::SmallVector<uint64_t> resultData;
    resultData.reserve(data.size());

    for (size_t limbIdx = 0; limbIdx < getNumLimbs(); ++limbIdx) {
      uint64_t modulus = moduli[limbIdx];
      for (size_t coeffIdx = 0; coeffIdx < numCoeffs; ++coeffIdx) {
        uint64_t a = getElement(limbIdx, coeffIdx);
        uint64_t b = other.getElement(limbIdx, coeffIdx);
        unsigned __int128 prod = (unsigned __int128)a * b;
        uint64_t res = prod % modulus;
        resultData.push_back(res);
      }
    }
    return RNSPolynomial(std::move(resultData), moduli, representation);
  }

  if (representation == Form::COEFF && other.representation == Form::COEFF) {
    return toNtt().mul(other.toNtt()).toCoefficient();
  }

  return RNSPolynomial();
}

RNSPolynomial RNSPolynomial::toNtt(rns::RNSAttr rootAttr) const {
  assert(representation == Form::COEFF && "Already in NTT representation");

  llvm::SmallVector<uint64_t> resultData;
  resultData.reserve(data.size());

  if (rootAttr) {
    assert(rootAttr.getValues().size() == getNumLimbs() &&
           "mismatch in number of limbs for root attribute");
  }

  for (size_t limbIdx = 0; limbIdx < getNumLimbs(); ++limbIdx) {
    uint64_t modulus = moduli[limbIdx];
    uint64_t rootOfUnity;

    if (rootAttr) {
      auto rootMA =
          llvm::cast<mod_arith::ModArithAttr>(rootAttr.getValues()[limbIdx]);
      rootOfUnity = rootMA.getValue().getValue().getZExtValue();
    } else {
      llvm::APInt qAp(64, modulus);
      std::optional<llvm::APInt> rootOpt =
          findPrimitive2nthRoot(qAp, numCoeffs);
      assert(rootOpt.has_value() && "Primitive 2n-th root of unity not found");
      rootOfUnity = rootOpt->getZExtValue();
    }

    std::vector<uint64_t> limbCoeffs;
    limbCoeffs.reserve(numCoeffs);
    for (size_t coeffIdx = 0; coeffIdx < numCoeffs; ++coeffIdx) {
      limbCoeffs.push_back(getElement(limbIdx, coeffIdx));
    }

    nttInPlace(limbCoeffs, modulus, rootOfUnity);
    resultData.insert(resultData.end(), limbCoeffs.begin(), limbCoeffs.end());
  }

  return RNSPolynomial(std::move(resultData), moduli, Form::EVAL);
}

RNSPolynomial RNSPolynomial::toCoefficient(rns::RNSAttr rootAttr) const {
  assert(representation == Form::EVAL &&
         "Already in Coefficient representation");

  llvm::SmallVector<uint64_t> resultData;
  resultData.reserve(data.size());

  if (rootAttr) {
    assert(rootAttr.getValues().size() == getNumLimbs() &&
           "mismatch in number of limbs for root attribute");
  }

  for (size_t limbIdx = 0; limbIdx < getNumLimbs(); ++limbIdx) {
    uint64_t modulus = moduli[limbIdx];
    uint64_t rootOfUnity;

    if (rootAttr) {
      auto rootMA =
          llvm::cast<mod_arith::ModArithAttr>(rootAttr.getValues()[limbIdx]);
      rootOfUnity = rootMA.getValue().getValue().getZExtValue();
    } else {
      llvm::APInt qAp(64, modulus);
      std::optional<llvm::APInt> rootOpt =
          findPrimitive2nthRoot(qAp, numCoeffs);
      assert(rootOpt.has_value() && "Primitive 2n-th root of unity not found");
      rootOfUnity = rootOpt->getZExtValue();
    }

    std::vector<uint64_t> limbCoeffs;
    limbCoeffs.reserve(numCoeffs);
    for (size_t coeffIdx = 0; coeffIdx < numCoeffs; ++coeffIdx) {
      limbCoeffs.push_back(getElement(limbIdx, coeffIdx));
    }

    inttInPlace(limbCoeffs, modulus, rootOfUnity);
    resultData.insert(resultData.end(), limbCoeffs.begin(), limbCoeffs.end());
  }

  return RNSPolynomial(std::move(resultData), moduli, Form::COEFF);
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
