#include "lib/Analysis/NoiseAnalysis/BGV/Noise.h"

#include <cmath>

#include "lib/Analysis/NoiseAnalysis/Params.h"
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

#define DEBUG_TYPE "BGVNoise"

namespace mlir {
namespace heir {
namespace bgv {

template <bool W>
std::string Noise<W>::toString() const {
  switch (noiseType) {
    case (NoiseType::UNINITIALIZED):
      return "Noise(uninitialized)";
    case (NoiseType::SET):
      return "Noise(" + std::to_string(log(getValue()) / log(2)) + ") ";
  }
}

template <bool W>
std::string Noise<W>::toBound(const LocalParam &param) const {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto bound = log(t * getValue()) / log(2);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << bound;
  return stream.str();
}

template <bool W>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Noise<W> &noise) {
  return os << noise.toString();
}

template <bool W>
Diagnostic &operator<<(Diagnostic &diagnostic, const Noise<W> &noise) {
  return diagnostic << noise.toString();
}

template <bool W>
double Noise<W>::getExpansionFactor(const LocalParam &param) {
  auto n = param.getSchemeParam()->getRingDim();
  if constexpr (W) {
    // worst-case
    return n;
  } else {
    // openfhe average case
    return 2.0 * sqrt(n);
  }
}

template <bool W>
double Noise<W>::getBoundErr(const LocalParam &param) {
  auto std0 = param.getSchemeParam()->getStd0();
  auto assurance = 6;
  auto boundErr = std0 * assurance;
  return boundErr;
}

template <bool W>
double Noise<W>::getBoundKey(const LocalParam &param) {
  auto boundKey = 1.0;
  return boundKey;
}

template <bool W>
Noise<W> Noise<W>::evalConstant(const LocalParam &param) {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  return Noise::of(t);
}

template <bool W>
Noise<W> Noise<W>::evalEncryptPk(const LocalParam &param) {
  auto boundErr = getBoundErr(param);
  auto boundKey = getBoundKey(param);
  auto expansionFactor = getExpansionFactor(param);

  double fresh = boundErr * (1. + 2. * expansionFactor * boundKey);
  return Noise::of(fresh);
}

template <bool W>
Noise<W> Noise<W>::evalAdd(const Noise &lhs, const Noise &rhs) {
  return Noise::of(lhs.getValue() + rhs.getValue() + 1);
}

template <bool W>
Noise<W> Noise<W>::evalMultNoRelin(const LocalParam &resultParam,
                                   const Noise<W> &lhs, const Noise<W> &rhs) {
  auto t = resultParam.getSchemeParam()->getPlaintextModulus();
  auto expansionFactor = getExpansionFactor(resultParam);

  return Noise::of((expansionFactor * t / 2) *
                   (lhs.getValue() * rhs.getValue() * 2 + lhs.getValue() +
                    rhs.getValue() + 1));
}

template <bool W>
Noise<W> Noise<W>::evalModReduce(const LocalParam &inputParam,
                                 const Noise<W> &input) {
  auto cv = inputParam.getDimension();
  assert(cv == 2);

  auto currentLogQi =
      inputParam.getSchemeParam()->getLogqi()[inputParam.getCurrentLevel()];

  double modulus = pow(2.0, currentLogQi);

  auto expansionFactor = getExpansionFactor(inputParam);
  auto boundKey = getBoundKey(inputParam);

  auto scaled = input.getValue() / modulus;
  auto added = (1.0 + expansionFactor * boundKey) / 2;
  return Noise::of(scaled + added);
}

// assume relinearize does not introduce error larger than mult
template <bool W>
Noise<W> Noise<W>::evalRelinearize(const LocalParam &inputParam,
                                   const Noise &input) {
  return input;
}

// assume rotation after mult...
// should be fixed later.
template <bool W>
Noise<W> Noise<W>::evalRotate(const LocalParam &inputParam,
                              const Noise &input) {
  return Noise::evalRelinearize(inputParam, input);
}

// instantiation
template class Noise<true>;
template class Noise<false>;

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
