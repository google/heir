#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYSYMBOLCOEFFMODEL_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYSYMBOLCOEFFMODEL_H_

#include <cassert>
#include <string>

#include "lib/Analysis/NoiseAnalysis/Symbolic.h"
#include "lib/Parameters/BGV/Params.h"

namespace mlir {
namespace heir {
namespace bgv {

// coefficient embedding noise model using variance
class NoiseBySymbolCoeffModel {
 public:
  // for MP24, NoiseState stores the variance var for the one coefficient of
  // critical quantity v = m + t * e, assuming coefficients are IID.
  //
  // MP24 states that for two polynomial multipication, the variance of one
  // coefficient of the result can be approximated by ringDim * var_0 * var_1,
  // because the polynomial multipication is a convolution.
  using StateType = noise::Expression;
  using SchemeParamType = SchemeParam;
  using LocalParamType = LocalParam;

 private:
  StateType evalEncryptPk(const LocalParamType &param, unsigned index) const;
  StateType evalEncryptSk(const LocalParamType &param, unsigned index) const;

 public:
  StateType evalEncrypt(const LocalParamType &param, unsigned index) const;
  StateType evalAdd(const StateType &lhs, const StateType &rhs) const;
  StateType evalMul(const LocalParamType &resultParam, const StateType &lhs,
                    const StateType &rhs) const;

  // logTotal: log(Ql / 2)
  // logBound: bound on ||m + t * e|| predicted by the model
  // logBudget: logTotal - logBound
  // as ||m + t * e|| < Ql / 2 for correct decryption
  double toLogBound(const LocalParamType &param, const StateType &noise) const;
  std::string toLogBoundString(const LocalParamType &param,
                               const StateType &noise) const;
  double toLogBudget(const LocalParamType &param, const StateType &noise) const;
  std::string toLogBudgetString(const LocalParamType &param,
                                const StateType &noise) const;
  double toLogTotal(const LocalParamType &param) const;
  std::string toLogTotalString(const LocalParamType &param) const;
};

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYSYMBOLCOEFFMODEL_H_
