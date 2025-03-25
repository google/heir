#include "lib/Analysis/NoiseAnalysis/CKKS/NoiseByVarianceCoeffModel.h"

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
namespace ckks {

// the formulae below are mainly taken from CCH+23
// "On the precision loss in approximate homomorphic encryption"
// https://ia.cr/2022/162 with modification that for message-noise
// multiplication we use worst-case bound.

using Model = NoiseByVarianceCoeffModel;

double Model::toLogBound(const LocalParamType &param, const StateType &noise) {
  // error probability 0.1%
  // though this only holds if every random variable is Gaussian
  // or similar to Gaussian
  // so this may give underestimation, see MP24 and CCH+23
  double alpha = 0.001;
  auto ringDim = param.getSchemeParam()->getRingDim();
  double bound =
      sqrt(2.0 * noise.getValue()) * erfinv(pow(1.0 - alpha, 1.0 / ringDim));
  return log2(bound);
  // rhoToRealError:
  // https://github.com/bencrts/CKKS_noise/blob/main/heuristics/CLT.py
  // double bound = sqrt(ringDim * noise.getValue()) *
  //               erfinv(pow(1.0 - alpha, 2.0 / ringDim));
  // return log2(bound) - param.getScale();
}

double Model::toLogBudget(const LocalParamType &param, const StateType &noise) {
  return toLogTotal(param) - toLogBound(param, noise);
}

double Model::toLogTotal(const LocalParamType &param) {
  double total = 0;
  auto logqi = param.getSchemeParam()->getLogqi();
  for (auto i = 0; i <= param.getSchemeParam()->getLevel(); ++i) {
    total += logqi[i];
  }
  return total - 1.0;
}

std::string Model::toLogBoundString(const LocalParamType &param,
                                    const StateType &noise) {
  auto logBound = toLogBound(param, noise);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logBound;
  return stream.str();
}

std::string Model::toLogBudgetString(const LocalParamType &param,
                                     const StateType &noise) {
  auto logBudget = toLogBudget(param, noise);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logBudget;
  return stream.str();
}

std::string Model::toLogTotalString(const LocalParamType &param) {
  auto logTotal = toLogTotal(param);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logTotal;
  return stream.str();
}

double Model::getVarianceErr(const LocalParamType &param) {
  auto std0 = param.getSchemeParam()->getStd0();
  return std0 * std0;
}

double Model::getVarianceKey(const LocalParamType &param) {
  // assume UNIFORM_TERNARY
  return 2.0 / 3.0;
}

typename Model::StateType Model::evalEncryptPk(const LocalParamType &param) {
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
  return StateType::of(fresh);
}

typename Model::StateType Model::evalEncryptSk(const LocalParamType &param) {
  auto varianceError = getVarianceErr(param);

  // secret key s
  // secret key encryption (-as + (q/t) * m + e, a)
  // v_fresh = e
  // var_fresh = var_error
  double fresh = varianceError;
  // max degree of 's' in v_fresh is 0.
  return StateType::of(fresh);
}

typename Model::StateType Model::evalEncrypt(const LocalParamType &param) {
  auto usePublicKey = param.getSchemeParam()->getUsePublicKey();
  // TODO
  auto isEncryptionTechniqueExtended = true;
  // param.getSchemeParam()->isEncryptionTechniqueExtended();
  if (isEncryptionTechniqueExtended) {
    // for extended encryption technique, namely encrypt at Qp then mod reduce
    // back to Q, the noise is modreduce(encrypt)
    auto ringDim = param.getSchemeParam()->getRingDim();
    auto varianceKey = getVarianceKey(param);
    // the error has the form tau_0 + tau_1 * s, where tau_i is uniform in
    // [-1/2, 1/2].
    auto added = (1.0 + ringDim * varianceKey) / 12.0;
    return StateType::of(added);
  }
  if (usePublicKey) {
    return evalEncryptPk(param);
  }
  return evalEncryptSk(param);
}

typename Model::StateType Model::evalConstant(const LocalParamType &param) {
  // constant is v = (q/t)m + 0
  return StateType::of(0);
}

typename Model::StateType Model::evalAdd(const StateType &lhs,
                                         const StateType &rhs) {
  // v_add = v_0 + v_1
  // assuming independence of course
  return StateType::of(lhs.getValue() + rhs.getValue());
}

typename Model::StateType Model::evalMul(const LocalParamType &resultParam,
                                         const LocalParamType &lhsParam,
                                         const LocalParamType &rhsParam,
                                         const StateType &lhs,
                                         const StateType &rhs) {
  auto ringDim = resultParam.getSchemeParam()->getRingDim();
  auto v0 = lhs.getValue();
  auto v1 = rhs.getValue();
  // v0 * v1 can be averaged to ringDim * v0 * v1
  double leftScale = pow(2.0, lhsParam.getScale());
  double rightScale = pow(2.0, rhsParam.getScale());
  // for v0 * m1, we do not have good method as m1 is _not uniformly random_
  // for m1 = Encode(z), we know the bound on ||z||_infty, then we have
  // ||m1||_infty <= Delta ||z||_infty, then we use worst-case bound
  // for v0 * m1 by
  //   N * ||m1||_infty * ||v0||_infty
  //     <= N * Delta * ||z||_infty * ||v0||_infty
  //
  // Expressing this in variance, as v0 can still be seen as uniformly random,
  // we can use the variance N^2 * Delta * 2 * ||z||_infty^2 * v0
  //
  // Still caution here as the coefficients of v0 are _not independent_.
  // It has noticeable distance from an independent one.
  // See "On the Independence Assumption in Quasi-Cyclic Code-Based
  // Cryptography"
  //
  // we currently assume z in [-1, 1], so ||z||_infty = 1
  // NOTE: when user is using z in, e.g. [-100, 100], we need to
  //   change here. Should rely on the plaintext backend to provide
  //   the correct bound.
  return StateType::of(ringDim * v0 * v1 +
                       ringDim * ringDim * v0 * rightScale * rightScale +
                       ringDim * ringDim * v1 * leftScale * leftScale);
}

typename Model::StateType Model::evalRelinearizeHYBRID(
    const LocalParamType &inputParam, const StateType &input) {
  // Ignore it for now, as often relinearization error is small
  return StateType::of(input.getValue());
}

typename Model::StateType Model::evalRelinearize(
    const LocalParamType &inputParam, const StateType &input) {
  // assume HYBRID
  // if we further introduce BV to SchemeParam we can have alternative
  // implementation.
  return evalRelinearizeHYBRID(inputParam, input);
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
