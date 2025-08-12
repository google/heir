#include "lib/Analysis/NoiseAnalysis/BGV/NoiseCanEmbModel.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <ios>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace mlir {
namespace heir {
namespace bgv {
// the formulae below are mainly taken from MMLGA22
// "Finding and Evaluating Parameters for BGV"
// https://eprint.iacr.org/2022/706

using Model = NoiseCanEmbModel;

double Model::toLogBound(const LocalParamType& param,
                         const StateType& noise) const {
  auto cm = getRingExpansionFactor(param);
  // ||a|| <= c_m * ||a||^{can}
  // noise.getValue stores log2(||a||^{can})
  return log2(cm) + noise.getValue();
}

double Model::toLogBudget(const LocalParamType& param,
                          const StateType& noise) const {
  return toLogTotal(param) - toLogBound(param, noise);
}

double Model::toLogTotal(const LocalParamType& param) const {
  double total = 0;
  auto logqi = param.getSchemeParam()->getLogqi();
  for (auto i = 0; i <= param.getCurrentLevel(); ++i) {
    total += logqi[i];
  }
  return total - 1.0;
}

double Model::getVarianceErr(const LocalParamType& param) const {
  auto std0 = param.getSchemeParam()->getStd0();
  return std0 * std0;
}

double Model::getVarianceKey(const LocalParamType& param) const {
  // assume UNIFORM_TERNARY
  return 2.0 / 3.0;
}

double Model::getRingExpansionFactor(const LocalParamType& param) const {
  [[maybe_unused]] auto N = param.getSchemeParam()->getRingDim();
  // Assert that N is a power of 2
  assert((N > 0) && ((N & (N - 1)) == 0) && "N must be a power of 2");
  // In power-of-two rings c_m = 1
  return 1.;
}

double Model::getAssuranceFactor(const LocalParamType& param) const {
  // probability that a exceeds its standard deviation by more than a factor of
  // D is roughly erfc(D) with erfc(6) = 2^-55, erfc(5) = 2^-40, erfc(4.5) =
  // 2^-32
  return 6.;
}

double Model::getBScale(const LocalParamType& param) const {
  auto varianceKey = getVarianceKey(param);
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto d = getAssuranceFactor(param);
  auto phi = getPhi(param);

  // B_scale = D * t * sqrt(phi(m)/12 * (1 + phi(m) * V_key)
  double innerTerm = (phi / 12.) * (1 + phi * varianceKey);
  return d * t * sqrt(innerTerm);
}

double Model::getBKs(const LocalParamType& param) const {
  auto varianceError = getVarianceErr(param);
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto d = getAssuranceFactor(param);
  auto phi = getPhi(param);

  // B_ks = D * t * phi(m) * sqrt(V_err / 12)
  return d * t * phi * sqrt(varianceError / 12.);
}

double Model::getPhi(const LocalParamType& param) const {
  return param.getSchemeParam()->getRingDim();
}

typename Model::StateType Model::evalEncryptPk(
    const LocalParamType& param) const {
  auto varianceError = getVarianceErr(param);
  // uniform ternary
  auto varianceKey = getVarianceKey(param);
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto d = getAssuranceFactor(param);
  auto phi = getPhi(param);

  // public key (-as + t * e, a)
  // public key encryption (-aus + t(u * e + e_0) + m, au + e_1)
  // ||m + t * (u * e + e_1 * s + e_0)||
  // <= D * t * sqrt(phi(m) * (1/12 + 2 * phi(m) * V_err * V_key + V_err))
  double innerTerm =
      phi * (1. / 12. + 2. * phi * varianceError * varianceKey + varianceKey);
  double fresh = d * t * sqrt(innerTerm);
  return StateType::of(fresh);
}

typename Model::StateType Model::evalEncryptSk(
    const LocalParamType& param) const {
  auto varianceError = getVarianceErr(param);
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto d = getAssuranceFactor(param);
  auto phi = getPhi(param);

  // secret key s
  // secret key encryption (-as + m + t * e, a)
  // ||m + t * e|| <= D * t * sqrt(phi(m) * (1/12 + V_err))
  double innerTerm = phi * (1. / 12. + varianceError);
  double fresh = d * t * sqrt(innerTerm);
  return StateType::of(fresh);
}

typename Model::StateType Model::evalEncrypt(
    const LocalParamType& param) const {
  auto usePublicKey = param.getSchemeParam()->getUsePublicKey();
  auto isEncryptionTechniqueExtended =
      param.getSchemeParam()->isEncryptionTechniqueExtended();
  if (isEncryptionTechniqueExtended) {
    // for extended encryption technique, namely encrypt at Qp then mod reduce
    // back to Q, the noise is modreduce(encrypt)
    return evalModReduce(param, evalEncryptPk(param));
  }
  if (usePublicKey) {
    return evalEncryptPk(param);
  }
  return evalEncryptSk(param);
}

typename Model::StateType Model::evalConstant(
    const LocalParamType& param) const {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto phi = getPhi(param);

  // noise part of the plaintext in a pt-ct multiplication
  // v_const <= t * sqrt(phi(m) / 12)
  return StateType::of(t * sqrt(phi / 12.0));
}

typename Model::StateType Model::evalAdd(const StateType& lhs,
                                         const StateType& rhs) const {
  // v_add <= v_0 + v_1
  return lhs + rhs;
}

typename Model::StateType Model::evalMul(const LocalParamType& resultParam,
                                         const StateType& lhs,
                                         const StateType& rhs) const {
  // v_mul <= v_0 * v_1
  return lhs * rhs;
}

typename Model::StateType Model::evalModReduce(const LocalParamType& inputParam,
                                               const StateType& input) const {
  auto currentLogqi =
      inputParam.getSchemeParam()->getLogqi()[inputParam.getCurrentLevel()];
  double modulus = pow(2.0, currentLogqi);

  // modulus switching is essentially a scaling operation
  // so the original error is scaled by the modulus
  // ||v_scaled|| = ||v_input|| / modulus
  auto scaled = input * (1. / modulus);
  // in the meantime, it will introduce a rounding error
  // (tau_0, tau_1) to (ct_0, ct_1)
  // ||tau_0 + tau_1 * s|| <= D * t * sqrt(phi(m)/12 * (1 + phi(m) * V_key) =
  // B_scale
  // ||v_ms|| <= ||v_scaled|| + B_scale
  auto bScale = StateType::of(getBScale(inputParam));
  return scaled + bScale;
}

typename Model::StateType Model::evalRelinearizeHYBRID(
    const LocalParamType& inputParam, const StateType& input) const {
  // for v_input, after modup and moddown, it remains the same (with rounding).
  // We only need to consider the error from key switching key
  // and rounding error during moddown.
  // Check section 3.2 of MMLGA22 for more details.
  auto dnum = inputParam.getSchemeParam()->getDnum();

  auto currentLevel = inputParam.getCurrentLevel();
  auto logpi = inputParam.getSchemeParam()->getLogpi();

  // TODO: prod of Pi() if Pi() is available instead of logPi()
  auto pi = inputParam.getSchemeParam()->getPi();
  double prodPi;
  double maxPi;
  size_t k;
  if (pi.empty()) {
    // values of pi are not set in schemeParam, so we use this
    std::vector<double> moduliPi(logpi.size());
    std::transform(logpi.begin(), logpi.end(), moduliPi.begin(),
                   [](double value) { return pow(2.0, value); });
    maxPi = *std::max_element(moduliPi.begin(), moduliPi.end());
    prodPi = std::accumulate(moduliPi.begin(), moduliPi.end(), 1.,
                             std::multiplies<double>());
    k = moduliPi.size();
  } else {
    // if real values of pi are set, we use those
    maxPi = *std::max_element(pi.begin(), pi.end());
    prodPi =
        std::accumulate(pi.begin(), pi.end(), 1., std::multiplies<double>());
    k = pi.size();
  }

  // v_ks = v + sqrt(dnum * (currentLevel + 1)) * p_l^(ceil(currentLevel / dnum)
  // * B_ks / P + sqrt(k) * B_scale
  double bKs = getBKs(inputParam);
  auto pPower = ceil(static_cast<double>(currentLevel) / dnum);
  auto noiseKs = StateType::of(
      sqrt(dnum * (currentLevel + 1)) *
      pow(static_cast<double>(maxPi), static_cast<double>(pPower)) * bKs /
      prodPi);

  double bScale = getBScale(inputParam);
  auto noiseScale = StateType::of(sqrt(k) * bScale);

  return input + noiseKs + noiseScale;
}

typename Model::StateType Model::evalRelinearize(
    const LocalParamType& inputParam, const StateType& input) const {
  // assume HYBRID
  // if we further introduce BV to SchemeParam we can have alternative
  // implementation.
  return evalRelinearizeHYBRID(inputParam, input);
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
