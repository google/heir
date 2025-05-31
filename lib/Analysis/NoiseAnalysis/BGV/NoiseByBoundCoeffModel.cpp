#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByBoundCoeffModel.h"

#include <cassert>
#include <cmath>

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace bgv {

// the formulae below are mainly taken from KPZ21
// "Revisiting Homomorphic Encryption Schemes for Finite Fields"
// https://ia.cr/2021/204

using LocalParamType = NoiseByBoundCoeffModel::LocalParamType;
using StateType = NoiseByBoundCoeffModel::StateType;
using SchemeParamType = NoiseByBoundCoeffModel::SchemeParamType;

double NoiseByBoundCoeffModel::toLogBound(const LocalParamType &param,
                                          const StateType &noise) const {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  // StateType only stores e in (m + t * e), so when we want to print the bound
  // we need to multiply t back.
  // noise.getValue is log2||e||
  return (noise * t).getValue();
}

double NoiseByBoundCoeffModel::toLogBudget(const LocalParamType &param,
                                           const StateType &noise) const {
  return toLogTotal(param) - toLogBound(param, noise);
}

double NoiseByBoundCoeffModel::toLogTotal(const LocalParamType &param) const {
  double total = 0;
  auto logqi = param.getSchemeParam()->getLogqi();
  for (auto i = 0; i <= param.getCurrentLevel(); ++i) {
    total += logqi[i];
  }
  return total - 1.0;
}

double NoiseByBoundCoeffModel::getExpansionFactor(
    const LocalParamType &param) const {
  auto n = param.getSchemeParam()->getRingDim();
  switch (variant) {
    case NoiseModelVariant::WORST_CASE:
      // well known from DPSZ12
      return n;
    case NoiseModelVariant::AVERAGE_CASE:
      // experimental result
      // cite HPS19 and KPZ21
      return 2.0 * sqrt(n);
    default:
      llvm_unreachable("Unknown noise model variant");
      return 0.0;
  }
}

double NoiseByBoundCoeffModel::getBoundErr(const LocalParamType &param) const {
  auto std0 = param.getSchemeParam()->getStd0();
  // probability of larger than 6 * std0 is less than 2^{-30}
  auto assurance = 6;
  auto boundErr = std0 * assurance;
  return boundErr;
}

double NoiseByBoundCoeffModel::getBoundKey(const LocalParamType &param) const {
  // assume UNIFORM_TERNARY
  auto boundKey = 1.0;
  return boundKey;
}

typename NoiseByBoundCoeffModel::StateType
NoiseByBoundCoeffModel::evalEncryptPk(const LocalParamType &param) const {
  auto boundErr = getBoundErr(param);
  auto boundKey = getBoundKey(param);
  auto expansionFactor = getExpansionFactor(param);

  // public key (-as + t * e, a)
  // public key encryption (-aus + t(u * e + e_0) + m, au + e_1)
  // m + t * (u * e + e_1 * s + e_0)
  // v_fresh = u * e + e_1 * s + e_0
  double fresh = boundErr * (1. + 2. * expansionFactor * boundKey);
  return StateType::of(fresh);
}

typename NoiseByBoundCoeffModel::StateType
NoiseByBoundCoeffModel::evalEncryptSk(const LocalParamType &param) const {
  auto boundErr = getBoundErr(param);

  // secret key s
  // secret key encryption (-as + m + t * e, a)
  // v_fresh = e
  double fresh = boundErr;
  return StateType::of(fresh);
}

typename NoiseByBoundCoeffModel::StateType NoiseByBoundCoeffModel::evalEncrypt(
    const LocalParamType &param) const {
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

typename NoiseByBoundCoeffModel::StateType NoiseByBoundCoeffModel::evalConstant(
    const LocalParamType &param) const {
  // constant is m + t * 0
  // v_constant = 0
  return StateType::of(0);
}

typename NoiseByBoundCoeffModel::StateType NoiseByBoundCoeffModel::evalAdd(
    const StateType &lhs, const StateType &rhs) const {
  // m_0 + tv_0 + m_1 + tv_1 <= [m_0 + m_1]_t + t(v_0 + v_1 + u)
  // v_add = v_0 + v_1 + u
  // where ||u|| <= 1
  return lhs + rhs + 1;
}

typename NoiseByBoundCoeffModel::StateType NoiseByBoundCoeffModel::evalMul(
    const LocalParamType &resultParam, const StateType &lhs,
    const StateType &rhs) const {
  auto t = resultParam.getSchemeParam()->getPlaintextModulus();
  auto expansionFactor = getExpansionFactor(resultParam);

  // (m_0 + tv_0) * (m_1 + tv_1) <=
  //   [m_0 * m_1]_t + t(v_0 * m_1 + v_1 * m_0 + v_0 * v_1 + r_m)
  // where m_0 * m_1 = [m_0 * m_1]_t + tr_m
  // ||r_m|| <= delta * t / 2, delta is the expansion factor
  // v_mul = v_0 * m_1 + v_1 * m_0 + v_0 * v_1 + r_m
  // ||v_mul|| <=
  //   (delta * t / 2) * (2 * ||v_0|| * ||v_1|| + ||v_0|| + ||v_1|| + 1)
  return (lhs * rhs * 2 + lhs + rhs + 1) * (expansionFactor * t / 2);
}

typename NoiseByBoundCoeffModel::StateType
NoiseByBoundCoeffModel::evalModReduce(const LocalParamType &inputParam,
                                      const StateType &input) const {
  // for cv > 2 the rounding error term is different!
  // like (tau_0, tau_1, tau_2) and the error becomes
  // tau_0 + tau_1 s + tau_2 s^2
  assert(inputParam.getDimension() == 2);

  auto currentLogqi =
      inputParam.getSchemeParam()->getLogqi()[inputParam.getCurrentLevel()];

  double modulus = pow(2.0, currentLogqi);

  auto expansionFactor = getExpansionFactor(inputParam);
  auto boundKey = getBoundKey(inputParam);

  // modulus switching is essentially a scaling operation
  // so the original error is scaled by the modulus
  // ||v_scaled|| = ||v_input|| / modulus
  auto scaled = input * (1.0 / modulus);
  // in the meantime, it will introduce an rounding error
  // (tau_0, tau_1) to the (ct_0, ct_1) where ||tau_i|| < t / 2
  // so ||tau_0 + tau_1 * s|| <= t / 2 (1 + delta ||s||)
  // ||v_added|| <= (1 + delta * Bkey) / 2
  // (1.0 + expansionFactor * boundKey) will give underestimation.
  auto added = StateType::of(1.0 + expansionFactor * boundKey);
  return scaled + added;
}

typename NoiseByBoundCoeffModel::StateType
NoiseByBoundCoeffModel::evalRelinearizeHYBRID(const LocalParamType &inputParam,
                                              const StateType &input) const {
  // for v_input, after modup and moddown, it remains the same (with rounding).
  // We only need to consider the error from key switching key
  // and rounding error during moddown.
  // Check the B.1.3 section of KPZ21 for more details.

  // also note that for cv > 3 (e.g. multiplication), we need to relinearize
  // more terms like ct_3 and ct_4.
  // this is a common path for mult relinearize and rotation relinearize
  // so no assertion here for now.

  auto dnum = inputParam.getSchemeParam()->getDnum();
  auto expansionFactor = getExpansionFactor(inputParam);
  auto boundErr = getBoundErr(inputParam);
  auto boundKey = getBoundKey(inputParam);

  auto currentLevel = inputParam.getCurrentLevel();
  // modup from Ql to QlP, so one more digit
  auto currentNumDigit = ceil(static_cast<double>(currentLevel + 1) / dnum) + 1;

  // log(qiq_{i+1}...), the digit size for a certain digit
  // we use log(pip_{i+1}...) as an approximation,
  // as we often choose P > each digit

  // the HYBRID key switching error is
  // sum over all digit (ct_2 * e_ksk)
  // there are "currentNumDigit" digits
  // and ||c_2|| <= digitSize / 2
  // ||c_2 * e_ksk|| <= delta * digitSize * Berr / 2
  // then moddown by P
  // so it becomes currentNumDigit * delta * Berr / 2
  auto scaled =
      StateType::of(currentNumDigit * expansionFactor * boundErr / 2.);

  // moddown added noise, similar to modreduce above.
  auto added = StateType::of((1.0 + expansionFactor * boundKey) / 2);

  // Get input + scaled + added
  // for relinearization after multiplication, often scaled + added is far less
  // than input.
  return input + scaled + added;
}

NoiseByBoundCoeffModel::StateType NoiseByBoundCoeffModel::evalRelinearize(
    const LocalParamType &inputParam, const StateType &input) const {
  // assume HYBRID
  // if we further introduce BV to SchemeParam we can have alternative
  // implementation.
  return evalRelinearizeHYBRID(inputParam, input);
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
