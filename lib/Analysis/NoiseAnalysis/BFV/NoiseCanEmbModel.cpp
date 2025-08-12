#include "lib/Analysis/NoiseAnalysis/BFV/NoiseCanEmbModel.h"

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
namespace bfv {

// canonical embedding noise model adapted
// from MMLGA22 https://eprint.iacr.org/2022/706
// from BMCM23 table 5 https://eprint.iacr.org/2023/600
// and from BGV/NoiseCanEmbModel

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
  auto d = getAssuranceFactor(param);
  auto phi = getPhi(param);

  // B_scale = D * sqrt(phi(m)/12 * (1 + phi(m) * V_key)
  double innerTerm = (phi / 12.) * (1 + phi * varianceKey);
  return d * sqrt(innerTerm);
}

double Model::getPhi(const LocalParamType& param) const {
  return param.getSchemeParam()->getRingDim();
}

typename Model::StateType Model::evalEncryptPk(
    const LocalParamType& param) const {
  auto varianceError = getVarianceErr(param);
  // uniform ternary
  auto varianceKey = getVarianceKey(param);
  auto d = getAssuranceFactor(param);
  auto phi = getPhi(param);

  // public key (-as + e, a)
  // public key encryption (-aus + u * e + e_0 + Delta * m, au + e_1)
  // ||u * e + e_1 * s + e_0||
  // <= D * sqrt(phi(m) * (2 * phi(m) * V_err * V_key + V_err))
  double innerTerm =
      phi * (2. * phi * varianceError * varianceKey + varianceKey);
  double fresh = d * sqrt(innerTerm);
  return StateType::of(fresh);
}

typename Model::StateType Model::evalEncryptSk(
    const LocalParamType& param) const {
  auto varianceError = getVarianceErr(param);
  auto d = getAssuranceFactor(param);
  auto phi = getPhi(param);

  // secret key s
  // secret key encryption (-as + Delta * m + e, a)
  // ||e|| <= D * sqrt(phi(m) * (V_err))
  double innerTerm = phi * (varianceError);
  double fresh = d * sqrt(innerTerm);
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
  // for canonical embedding, ||e1 * e2||^can <= ||e1||^can * ||e2||^can
  auto v0 = lhs;
  auto v1 = rhs;
  auto phi = getPhi(resultParam);
  auto t = resultParam.getSchemeParam()->getPlaintextModulus();
  auto logqi = resultParam.getSchemeParam()->getLogqi();
  auto logQ = std::accumulate(logqi.begin(), logqi.end(), 0.0);
  // we hope double is big enough...
  // if logQ > 1024... we may have a problem
  auto Q = pow(2.0, logQ);
  auto d = getAssuranceFactor(resultParam);

  // c0 + c1s = (Q/t)m + v + hQ
  // h is the high term, and has form (c1s + c0 - (Q/t)m - v)/Q
  // so its variance is (phi * Vkey + 4) / 12
  // B_h = D * sqrt(phi(m)/12 * (4 + phi(m) * V_key)
  auto varianceKey = getVarianceKey(resultParam);
  double innerH = (phi / 12.) * (1 + phi * varianceKey);
  auto bH = d * sqrt(innerH);

  // The rounding error is ignorable
  // Bscale could not be used here as it has the form
  // delta0 + delta1 * s + delta2 * s^2

  // See KPZ21
  auto term1 = v0 * v1 * t * (1. / Q);
  // See also BMCM23 Table 5
  auto term2 = (v0 + v1) * t * bH;
  return term1 + term2;
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
  // ||tau_0 + tau_1 * s|| <= D * sqrt(phi(m)/12 * (1 + phi(m) * V_key) =
  // B_scale
  // ||v_ms|| <= ||v_scaled|| + B_scale
  auto bScale = StateType::of(getBScale(inputParam));
  return scaled + bScale;
}

typename Model::StateType Model::evalRelinearizeHYBRID(
    const LocalParamType& inputParam, const StateType& input) const {
  // assume HYBRID and noise is negligible
  // compared with noise of multiplication
  // TODO: assert this happens after multiplication
  return input;
}

typename Model::StateType Model::evalRelinearize(
    const LocalParamType& inputParam, const StateType& input) const {
  // assume HYBRID
  // if we further introduce BV to SchemeParam we can have alternative
  // implementation.
  return evalRelinearizeHYBRID(inputParam, input);
}

}  // namespace bfv
}  // namespace heir
}  // namespace mlir
