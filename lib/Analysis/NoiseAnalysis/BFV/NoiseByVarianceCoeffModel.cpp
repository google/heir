#include "lib/Analysis/NoiseAnalysis/BFV/NoiseByVarianceCoeffModel.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <ios>
#include <numeric>
#include <sstream>
#include <string>

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Utils/MathUtils.h"

namespace mlir {
namespace heir {
namespace bfv {

// the formulae below are mainly taken from BMCM23
// "Improving and Automating BFV Parameters Selection: An Average-Case Approach"
// https://ia.cr/2023/600
// with modification that it works on Var(e) instead of invariant noise
// Reader may also refer to KPZ21
// "Revisiting Homomorphic Encryption Schemes for Finite Fields"
// to see how the formulae there are adapted to this model

double NoiseByVarianceCoeffModel::toLogBound(const LocalParamType& param,
                                             const StateType& noise) const {
  // error probability 2^-32
  // though this only holds if every random variable is Gaussian
  // or similar to Gaussian
  // so this may give underestimation, see MP24 and CCH+23
  double alpha = std::exp2(-32);
  auto ringDim = param.getSchemeParam()->getRingDim();
  // noise.getValue is log2(Var)
  double bound = (1. / 2) * (1 + noise.getValue()) +
                 log2(erfinv(pow(1.0 - alpha, 1.0 / ringDim)));
  return bound;
}

double NoiseByVarianceCoeffModel::toLogBudget(const LocalParamType& param,
                                              const StateType& noise) const {
  return toLogTotal(param) - toLogBound(param, noise);
}

double NoiseByVarianceCoeffModel::toLogTotal(
    const LocalParamType& param) const {
  double total = 0;
  auto logqi = param.getSchemeParam()->getLogqi();
  for (auto i = 0; i <= param.getSchemeParam()->getLevel(); ++i) {
    total += logqi[i];
  }
  double logT = log2(param.getSchemeParam()->getPlaintextModulus());
  return total - logT - 1.0;
}

double NoiseByVarianceCoeffModel::getVarianceErr(
    const LocalParamType& param) const {
  auto std0 = param.getSchemeParam()->getStd0();
  return std0 * std0;
}

double NoiseByVarianceCoeffModel::getVarianceKey(
    const LocalParamType& param) const {
  // assume UNIFORM_TERNARY
  return 2.0 / 3.0;
}

typename NoiseByVarianceCoeffModel::StateType
NoiseByVarianceCoeffModel::evalEncryptPk(const LocalParamType& param) const {
  auto varianceError = getVarianceErr(param);
  // uniform ternary
  auto varianceKey = getVarianceKey(param);
  auto n = param.getSchemeParam()->getRingDim();
  // public key (-as + e, a)
  // public key encryption (-aus + (u * e + e_0) + (q/t) * m, au + e_1)
  // v_fresh = u * e + e_1 * s + e_0
  // var_fresh = (2n * var_key + 1) * var_error
  // for ringDim, see header comment for explanation
  double fresh = varianceError * (2. * n * varianceKey + 1.);
  // max degree of 's' in v_fresh is 1.
  return StateType::of(fresh, 1);
}

typename NoiseByVarianceCoeffModel::StateType
NoiseByVarianceCoeffModel::evalEncryptSk(const LocalParamType& param) const {
  auto varianceError = getVarianceErr(param);

  // secret key s
  // secret key encryption (-as + (q/t) * m + e, a)
  // v_fresh = e
  // var_fresh = var_error
  double fresh = varianceError;
  // max degree of 's' in v_fresh is 0.
  return StateType::of(fresh, 0);
}

typename NoiseByVarianceCoeffModel::StateType
NoiseByVarianceCoeffModel::evalEncrypt(const LocalParamType& param) const {
  auto usePublicKey = param.getSchemeParam()->getUsePublicKey();
  auto isEncryptionTechniqueExtended =
      param.getSchemeParam()->isEncryptionTechniqueExtended();
  if (isEncryptionTechniqueExtended) {
    // for extended encryption technique, namely encrypt at Qp then mod reduce
    // back to Q, the noise is modreduce(encrypt)
    auto ringDim = param.getSchemeParam()->getRingDim();
    auto varianceKey = getVarianceKey(param);
    // the error has the form tau_0 + tau_1 * s, where tau_i is uniform in
    // [-1/2, 1/2].
    auto added = (1.0 + ringDim * varianceKey) / 12.0;
    // max degree of 's' is 1.
    return StateType::of(added, 1);
  }
  if (usePublicKey) {
    return evalEncryptPk(param);
  }
  return evalEncryptSk(param);
}

typename NoiseByVarianceCoeffModel::StateType
NoiseByVarianceCoeffModel::evalConstant(const LocalParamType& param) const {
  // constant is v = (q/t)m + 0
  return StateType::of(0, 0);
}

typename NoiseByVarianceCoeffModel::StateType
NoiseByVarianceCoeffModel::evalAdd(const StateType& lhs,
                                   const StateType& rhs) const {
  // v_add = v_0 + v_1
  // assuming independence of course
  // max degree of 's' in v_add is max(d_0, d_1)
  return lhs + rhs;
}

typename NoiseByVarianceCoeffModel::StateType
NoiseByVarianceCoeffModel::evalMul(const LocalParamType& resultParam,
                                   const StateType& lhs,
                                   const StateType& rhs) const {
  auto ringDim = resultParam.getSchemeParam()->getRingDim();
  auto v0 = lhs;
  auto d0 = lhs.getDegree();
  auto v1 = rhs;
  auto d1 = rhs.getDegree();
  auto newDegree = std::max(d0, d1);
  auto t = resultParam.getSchemeParam()->getPlaintextModulus();

  auto logqi = resultParam.getSchemeParam()->getLogqi();
  auto logQ = std::accumulate(logqi.begin(), logqi.end(), 0.0);
  // we hope double is big enough...
  // if logQ > 1024... we may have a problem
  auto Q = pow(2.0, logQ);

  auto varianceKey = getVarianceKey(resultParam);

  // ((q/t)m_0 + e_0 + k_0 * q) * ((q/t)m_1 + e_1 + k_1 * q)
  // = (q/t)^2 m_0 * m_1
  //   + (q/t)(m_0 * e_1 + m_1 * e_0
  //      + t(e_0 * k_1 + e_1 * k_0) + (t/q)e_0 * e_1)
  //   + some q^2/t
  //
  // after scaling by (q/t) we get
  //
  // (q/t) m_0 * m_1
  //    + (m_0 * e_1 + m_1 * e_0 +
  //    + t(e_0 * k_1 + e_1 * k_0) + (t/q)e_0 * e_1)
  //    + some q
  //
  // v_mul = v_0 * v_1 * (t/q)
  //    + v_0 * m_1 + v_1 * m_0
  //    + t(v_0 * k_1 + v_1 * k_0)
  //
  // k_i * Q is the not rounded part when we compute c(s) = c_0 + c_1 * s
  // so we have k_i = (c(s) - round(c(s))) / Q
  // so _heuristically_ we have k_i = tau_0 + tau_1 * s
  // where tau_i is uniformly in [-1/2, 1/2]
  // then Var(k_i) = 1/12(1 + ringDim * var_key)
  auto term1 = (v0 * v1 * ringDim) * t * t * (1. / Q / Q);
  auto term2 = (v0 + v1) * ringDim * t * t * (1. / 12.);
  auto zero = StateType::of(0, 0);
  auto term3 = zero;
  // only ct-ct mul has this term
  // as pt does not have k_i * Q
  if (v0.withNewDegree(0) != zero && v1.withNewDegree(0) != zero) {
    // Key part of BMCM23 is that, for v_0 * k_1 where v_0 is of degree d0
    // and k_1 is of degree 1, the degree of the resulting term is d0 + 1.
    // Then we need a correction factor f(d0 + 1) multiplied to v_0.
    // BMCM has an approximation for it, but for low degree we can just
    // use f(d0 + 1) = d0 + 1.
    // CAUTION: this formula won't work for high degree like 20, which means
    // the circuit is 20-level deep.
    term3 = (v0 * (d0 + 1) + v1 * (d1 + 1)) * (1 + ringDim * varianceKey) *
            (1. / 12.) * ringDim * t * t;
    // the degree of the resulting term is max(d0 + 1, d1 + 1)
    newDegree += 1;
  }
  return (term1 + term2 + term3).withNewDegree(newDegree);
}

typename NoiseByVarianceCoeffModel::StateType
NoiseByVarianceCoeffModel::evalRelinearizeHYBRID(
    const LocalParamType& inputParam, const StateType& input) const {
  // for v_input, after modup and moddown, it remains the same (with rounding).
  // We only need to consider the error from key switching key
  // and rounding error during moddown.
  // Check the B.1.3 section of KPZ21 for more details.

  // also note that for cv > 3 (e.g. multiplication), we need to relinearize
  // more terms like ct_3 and ct_4.
  // this is a common path for mult relinearize and rotation relinearize
  // so no assertion here for now.

  auto dnum = inputParam.getSchemeParam()->getDnum();
  auto varianceErr = getVarianceErr(inputParam);
  auto varianceKey = getVarianceKey(inputParam);
  auto ringDim = inputParam.getSchemeParam()->getRingDim();
  auto t = inputParam.getSchemeParam()->getPlaintextModulus();

  auto level = inputParam.getSchemeParam()->getLevel();
  // modup from Q to QP, so one more digit
  auto numDigit = ceil(static_cast<double>(level + 1) / dnum) + 1;

  // log(qiq_{i+1}...), the digit size for a certain digit
  // we use log(pip_{i+1}...) as an approximation,
  // as we often choose P > each digit

  // the error for HYBRID key switching error is
  // t * sum over all digit (ct_2 * e_ksk)
  // there are "numDigit" digits
  // and c_2 uniformly from [-digitSize / 2, digitSize / 2]
  // for ringDim, see header comment for explanation
  // moddown by P, as digitSize / P < 1, we directly use 1/12
  auto scaled =
      NoiseState::of(t * t * numDigit * (1.0 / 12.0) * ringDim * varianceErr);

  // Some papers just say hey we mod down by P so the error added is just mod
  // reduce, but the error for mod reduce is different for approximate mod down.
  // Anyway, this term is not the major term.
  // moddown added noise, similar to modreduce.
  auto added = NoiseState::of((1.0 + ringDim * varianceKey) / 12.0);

  // for relinearization after multiplication, often scaled + added is far less
  // than input.
  return input + scaled + added;
}

typename NoiseByVarianceCoeffModel::StateType
NoiseByVarianceCoeffModel::evalRelinearize(const LocalParamType& inputParam,
                                           const StateType& input) const {
  // assume HYBRID
  // if we further introduce BV to SchemeParam we can have alternative
  // implementation.
  return evalRelinearizeHYBRID(inputParam, input);
}

}  // namespace bfv
}  // namespace heir
}  // namespace mlir
