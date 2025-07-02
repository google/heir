#ifndef LIB_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYVARIANCECOEFFMODEL_H_
#define LIB_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYVARIANCECOEFFMODEL_H_

#include <cassert>
#include <string>

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Parameters/BGV/Params.h"

namespace mlir {
namespace heir {
namespace bgv {

// coefficient embedding noise model using variance
class NoiseByVarianceCoeffModel {
 public:
  // for MP24, NoiseState stores the variance log2(var) for the one coefficient
  // of critical quantity v = m + t * e, assuming coefficients are IID.
  //
  // MP24 states that for two polynomial multipication, the variance of one
  // coefficient of the result can be approximated by ringDim * var_0 * var_1,
  // because the polynomial multipication is a convolution.
  using StateType = NoiseState;
  using SchemeParamType = SchemeParam;
  using LocalParamType = LocalParam;

 private:
  double getVarianceErr(const LocalParamType &param) const;
  double getVarianceKey(const LocalParamType &param) const;

  StateType evalEncryptPk(const LocalParamType &param) const;
  StateType evalEncryptSk(const LocalParamType &param) const;
  StateType evalRelinearizeHYBRID(const LocalParamType &inputParam,
                                  const StateType &input) const;

 public:
  StateType evalEncrypt(const LocalParamType &param) const;
  StateType evalConstant(const LocalParamType &param) const;
  StateType evalAdd(const StateType &lhs, const StateType &rhs) const;
  StateType evalMul(const LocalParamType &resultParam, const StateType &lhs,
                    const StateType &rhs) const;
  StateType evalRelinearize(const LocalParamType &inputParam,
                            const StateType &input) const;
  StateType evalModReduce(const LocalParamType &inputParam,
                          const StateType &input) const;

  // logTotal: log(Ql / 2)
  // logBound: bound on ||m + t * e|| predicted by the model
  // logBudget: logTotal - logBound
  // as ||m + t * e|| < Ql / 2 for correct decryption
  double toLogBound(const LocalParamType &param, const StateType &noise) const;
  double toLogBudget(const LocalParamType &param, const StateType &noise) const;
  double toLogTotal(const LocalParamType &param) const;
};

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYVARIANCECOEFFMODEL_H_
