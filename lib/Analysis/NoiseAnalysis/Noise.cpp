#include "lib/Analysis/NoiseAnalysis/Noise.h"

#include <cmath>

#include "lib/Analysis/NoiseAnalysis/Params.h"
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

#define DEBUG_TYPE "Noise"

namespace mlir {
namespace heir {

std::string Noise::toString() const {
  switch (noiseType) {
    case (NoiseType::UNINITIALIZED):
      return "Noise(uninitialized)";
    case (NoiseType::SET):
      return "Noise(" + std::to_string(log(getValue()) / log(2)) + ") ";
  }
}

std::string Noise::toBound(const LocalParam &param) const {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto bound = log(t * getValue()) / log(2);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << bound;
  return stream.str();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Noise &variance) {
  return os << variance.toString();
}

Diagnostic &operator<<(Diagnostic &diagnostic, const Noise &variance) {
  return diagnostic << variance.toString();
}

double Noise::getExpansionFactor(const LocalParam &param) {
  auto n = param.getSchemeParam()->getRingDim();
  // openfhe average case
  auto expansionFactor = 2.0 * sqrt(n);
  // worst-case
  // auto expansionFactor = n;
  return expansionFactor;
}

double Noise::getBoundErr(const LocalParam &param) {
  auto std0 = param.getSchemeParam()->getStd0();
  auto assurance = 6;
  auto boundErr = std0 * assurance;
  return boundErr;
}

double Noise::getBoundKey(const LocalParam &param) {
  auto boundKey = 1.0;
  return boundKey;
}

Noise Noise::evalConstant(const LocalParam &param) {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  return Noise::of(t);
}

Noise Noise::evalEncryptPk(const LocalParam &param) {
  auto boundErr = getBoundErr(param);
  auto boundKey = getBoundKey(param);
  auto expansionFactor = getExpansionFactor(param);

  double fresh = boundErr * (1. + 2. * expansionFactor * boundKey);
  return Noise::of(fresh);
}

Noise Noise::evalAdd(const Noise &lhs, const Noise &rhs) {
  return Noise::of(lhs.getValue() + rhs.getValue() + 1);
}
Noise Noise::evalMultNoRelin(const LocalParam &resultParam, const Noise &lhs,
                             const Noise &rhs) {
  auto t = resultParam.getSchemeParam()->getPlaintextModulus();
  auto expansionFactor = getExpansionFactor(resultParam);

  return Noise::of((expansionFactor * t / 2) *
                   (lhs.getValue() * rhs.getValue() * 2 + lhs.getValue() +
                    rhs.getValue() + 1));
}

Noise Noise::evalModReduce(const LocalParam &inputParam, const Noise &input) {
  auto cv = inputParam.getDimension();
  assert(cv == 2);
  double modulus =
      1L << inputParam.getSchemeParam()->getQi()[inputParam.getCurrentLevel()];

  auto expansionFactor = getExpansionFactor(inputParam);
  auto boundKey = getBoundKey(inputParam);

  auto scaled = input.getValue() / modulus;
  auto added = (1.0 + expansionFactor * boundKey) / 2;
  return Noise::of(scaled + added);
}

Noise Noise::evalRelinearize(const LocalParam &inputParam, const Noise &input) {
  return input;
}

Noise Noise::evalRotate(const LocalParam &inputParam, const Noise &input) {
  return Noise::evalRelinearize(inputParam, input);
}

}  // namespace heir
}  // namespace mlir
