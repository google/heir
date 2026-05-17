#include "lib/Analysis/NoiseAnalysis/BGV/NoiseBySymbolCoeffModel.h"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <ios>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include "lib/Utils/MathUtils.h"

namespace mlir {
namespace heir {
namespace bgv {

using namespace mlir::heir::noise;

double NoiseBySymbolCoeffModel::toLogBound(const LocalParamType &param,
                                           const StateType &noise) const {
  double alpha = pow(2.0, -32);
  auto ringDim = param.getSchemeParam()->getRingDim();
  // variance is log2(Var(e))
  auto variance = noise.getVariance(ringDim);
  double bound = (1. / 2.) * (1 + variance) +
                 log2(erfinv(pow(1.0 - alpha, 1.0 / ringDim)));
  return bound;
}

double NoiseBySymbolCoeffModel::toLogBudget(const LocalParamType &param,
                                            const StateType &noise) const {
  return toLogTotal(param) - toLogBound(param, noise);
}

double NoiseBySymbolCoeffModel::toLogTotal(const LocalParamType &param) const {
  double total = 0;
  auto logqi = param.getSchemeParam()->getLogqi();
  for (auto i = 0; i <= param.getCurrentLevel(); ++i) {
    total += logqi[i];
  }
  return total - 1.0;
}

std::string NoiseBySymbolCoeffModel::toLogBoundString(
    const LocalParamType &param, const StateType &noise) const {
  auto logBound = toLogBound(param, noise);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logBound;
  return stream.str();
}

std::string NoiseBySymbolCoeffModel::toLogBudgetString(
    const LocalParamType &param, const StateType &noise) const {
  auto logBudget = toLogBudget(param, noise);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logBudget;
  return stream.str();
}

std::string NoiseBySymbolCoeffModel::toLogTotalString(
    const LocalParamType &param) const {
  auto logTotal = toLogTotal(param);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logTotal;
  return stream.str();
}

typename NoiseBySymbolCoeffModel::StateType
NoiseBySymbolCoeffModel::evalEncryptPk(const LocalParamType &param,
                                       unsigned index) const {
  auto stderr = param.getSchemeParam()->getStd0();
  Expr ei = Symbol(Symbol::GAUSSIAN, "e" + std::to_string(index), stderr);
  Expr s = Symbol(Symbol::UNIFORM_TERNARY, "s", 0);
  Expr epk = Symbol(Symbol::GAUSSIAN, "epk", stderr);
  Expr ui = Symbol(Symbol::UNIFORM_TERNARY, "u" + std::to_string(index), 0);

  Expr t = Symbol(Symbol::CONSTANT, "t",
                  param.getSchemeParam()->getPlaintextModulus());
  return t * (ei * s + epk * ui);
}

typename NoiseBySymbolCoeffModel::StateType
NoiseBySymbolCoeffModel::evalEncryptSk(const LocalParamType &param,
                                       unsigned index) const {
  auto stderr = param.getSchemeParam()->getStd0();
  Expr ei = Symbol(Symbol::GAUSSIAN, "e" + std::to_string(index), stderr);

  Expr t = Symbol(Symbol::CONSTANT, "t",
                  param.getSchemeParam()->getPlaintextModulus());
  return t * ei;
}

typename NoiseBySymbolCoeffModel::StateType
NoiseBySymbolCoeffModel::evalEncrypt(const LocalParamType &param,
                                     unsigned index) const {
  auto usePublicKey = param.getSchemeParam()->getUsePublicKey();
  if (usePublicKey) {
    return evalEncryptPk(param, index);
  }
  return evalEncryptSk(param, index);
}

typename NoiseBySymbolCoeffModel::StateType NoiseBySymbolCoeffModel::evalMul(
    const LocalParamType &resultParam, const StateType &lhs,
    const StateType &rhs) const {
  return lhs * rhs;
}

typename NoiseBySymbolCoeffModel::StateType NoiseBySymbolCoeffModel::evalAdd(
    const StateType &lhs, const StateType &rhs) const {
  return lhs + rhs;
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
