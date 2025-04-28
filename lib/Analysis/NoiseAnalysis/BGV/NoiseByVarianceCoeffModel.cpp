#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByVarianceCoeffModel.h"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <ios>
#include <numeric>
#include <sstream>
#include <string>

#include "lib/Utils/MathUtils.h"

namespace mlir {
namespace heir {
namespace bgv {

// the formulae below are mainly taken from MP24
// "A Central Limit Framework for Ring-LWE Noise Analysis"
// https://ia.cr/2019/452
// and CLP23
// "Optimisations and tradeoffs for HElib"
// https://ia.cr/2023/104

using Model = NoiseByVarianceCoeffModel;

double Model::toLogBound(const LocalParamType &param,
                         const StateType &noise) const {
  // error probability 0.1%
  // though this only holds if every random variable is Gaussian
  // or similar to Gaussian
  // so this may give underestimation, see MP24 and CCH+23
  double alpha = 0.001;
  auto ringDim = param.getSchemeParam()->getRingDim();
  double bound =
      sqrt(2.0 * noise.getValue()) * erfinv(pow(1.0 - alpha, 1.0 / ringDim));
  return log2(bound);
}

double Model::toLogBudget(const LocalParamType &param,
                          const StateType &noise) const {
  return toLogTotal(param) - toLogBound(param, noise);
}

double Model::toLogTotal(const LocalParamType &param) const {
  double total = 0;
  auto logqi = param.getSchemeParam()->getLogqi();
  for (auto i = 0; i <= param.getCurrentLevel(); ++i) {
    total += logqi[i];
  }
  return total - 1.0;
}

std::string Model::toLogBoundString(const LocalParamType &param,
                                    const StateType &noise) const {
  auto logBound = toLogBound(param, noise);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logBound;
  return stream.str();
}

std::string Model::toLogBudgetString(const LocalParamType &param,
                                     const StateType &noise) const {
  auto logBudget = toLogBudget(param, noise);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logBudget;
  return stream.str();
}

std::string Model::toLogTotalString(const LocalParamType &param) const {
  auto logTotal = toLogTotal(param);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logTotal;
  return stream.str();
}

double Model::getVarianceErr(const LocalParamType &param) const {
  auto std0 = param.getSchemeParam()->getStd0();
  return std0 * std0;
}

double Model::getVarianceKey(const LocalParamType &param) const {
  // assume UNIFORM_TERNARY
  return 2.0 / 3.0;
}

typename Model::StateType Model::evalEncryptPk(
    const LocalParamType &param) const {
  auto varianceError = getVarianceErr(param);
  // uniform ternary
  auto varianceKey = getVarianceKey(param);
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto n = param.getSchemeParam()->getRingDim();
  // public key (-as + t * e, a)
  // public key encryption (-aus + t(u * e + e_0) + m, au + e_1)
  // v_fresh = m + t * (u * e + e_1 * s + e_0)
  // var_fresh = t^2 * (2n * var_key + 1) * var_error
  // for ringDim, see header comment for explanation
  double fresh = t * t * varianceError * (2. * n * varianceKey + 1.);
  return StateType::of(fresh);
}

typename Model::StateType Model::evalEncryptSk(
    const LocalParamType &param) const {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto varianceError = getVarianceErr(param);

  // secret key s
  // secret key encryption (-as + m + t * e, a)
  // v_fresh = t * e
  // var_fresh = t^2 * var_error
  double fresh = t * t * varianceError;
  return StateType::of(fresh);
}

typename Model::StateType Model::evalEncrypt(
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

typename Model::StateType Model::evalConstant(
    const LocalParamType &param) const {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  // constant is v = m + t * 0
  // assume m is uniform from [-t/2, t/2]
  // var_constant = t * t / 12
  return StateType::of(t * t / 12.0);
}

typename Model::StateType Model::evalAdd(const StateType &lhs,
                                         const StateType &rhs) const {
  // v_add = v_0 + v_1
  // assuming independent of course
  return StateType::of(lhs.getValue() + rhs.getValue());
}

typename Model::StateType Model::evalMul(const LocalParamType &resultParam,
                                         const StateType &lhs,
                                         const StateType &rhs) const {
  auto ringDim = resultParam.getSchemeParam()->getRingDim();
  auto v0 = lhs.getValue();
  auto v1 = rhs.getValue();

  // v_mul = v_0 * v_1
  // for ringDim, see header comment for explanation
  return StateType::of(ringDim * v0 * v1);
}

typename Model::StateType Model::evalModReduce(const LocalParamType &inputParam,
                                               const StateType &input) const {
  [[maybe_unused]] auto cv = inputParam.getDimension();
  // for cv > 2 the rounding error term is different!
  // like (tau_0, tau_1, tau_2) and the error becomes
  // tau_0 + tau_1 s + tau_2 s^2
  assert(cv == 2);

  auto currentLogqi =
      inputParam.getSchemeParam()->getLogqi()[inputParam.getCurrentLevel()];

  double modulus = pow(2.0, currentLogqi);

  auto ringDim = inputParam.getSchemeParam()->getRingDim();
  auto varianceKey = getVarianceKey(inputParam);

  // modulus switching is essentially a scaling operation
  // so the original error is scaled by the modulus
  // v_scaled = v_input / modulus
  // var_scaled = var_input / (modulus * modulus)
  auto scaled = input.getValue() / (modulus * modulus);
  // in the meantime, it will introduce an rounding error
  // (tau_0, tau_1) to the (ct_0, ct_1) where ||tau_i|| < t / 2
  // so tau_0 + tau_1 * s has the variance
  // var_added = var_const * (1.0 + var_key * ringDim)
  auto varianceConst = evalConstant(inputParam).getValue();
  auto added = varianceConst * (1.0 + ringDim * varianceKey);
  return StateType::of(scaled + added);
}

typename Model::StateType Model::evalRelinearizeHYBRID(
    const LocalParamType &inputParam, const StateType &input) const {
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

  auto currentLevel = inputParam.getCurrentLevel();
  // modup from Ql to QlP, so one more digit
  auto currentNumDigit = ceil(static_cast<double>(currentLevel + 1) / dnum) + 1;

  // log(qiq_{i+1}...), the digit size for a certain digit
  // we use log(pip_{i+1}...) as an approximation,
  // as we often choose P > each digit
  auto logqi = inputParam.getSchemeParam()->getLogqi();
  auto logDigitSize = std::accumulate(logqi.begin(), logqi.end(), 0.0);
  // omega in literature
  auto digitSize = pow(2.0, logDigitSize);

  // the critical quantity for HYBRID key switching error is
  // t * sum over all digit (ct_2 * e_ksk)
  // there are "currentNumDigit" digits
  // and c_2 uniformly from [-digitSize / 2, digitSize / 2]
  // for ringDim, see header comment for explanation
  auto varianceKeySwitch = t * t * currentNumDigit *
                           (digitSize * digitSize / 12.0) * ringDim *
                           varianceErr;

  // moddown by P
  auto scaled = varianceKeySwitch / (digitSize * digitSize);

  // Some papers just say hey we mod down by P so the error added is just mod
  // reduce, but the error for mod reduce is different for approximate mod down.
  // Anyway, this term is not the major term.
  auto varianceConst = evalConstant(inputParam).getValue();
  auto added = varianceConst * (1.0 + ringDim * varianceKey);

  // for relinearization after multiplication, often scaled + added is far less
  // than input.
  return StateType::of(input.getValue() + scaled + added);
}

typename Model::StateType Model::evalRelinearize(
    const LocalParamType &inputParam, const StateType &input) const {
  // assume HYBRID
  // if we further introduce BV to SchemeParam we can have alternative
  // implementation.
  return evalRelinearizeHYBRID(inputParam, input);
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
