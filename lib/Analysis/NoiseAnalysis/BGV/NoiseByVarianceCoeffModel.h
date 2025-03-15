#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYVARIANCECOEFFMODEL_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYVARIANCECOEFFMODEL_H_

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
  // for MP24, NoiseState stores the variance var for the one coefficient of
  // critical quantity v = m + t * e, assuming coefficients are IID.
  //
  // MP24 states that for two polynomial multipication, the variance of one
  // coefficient of the result can be approximated by ringDim * var_0 * var_1,
  // because the polynomial multipication is a convolution.
  using StateType = NoiseState;
  using SchemeParamType = SchemeParam;
  using LocalParamType = LocalParam;

 private:
  static double getVarianceErr(const LocalParamType &param);
  static double getVarianceKey(const LocalParamType &param);

  static StateType evalEncryptPk(const LocalParamType &param);
  static StateType evalEncryptSk(const LocalParamType &param);
  static StateType evalRelinearizeHYBRID(const LocalParamType &inputParam,
                                         const StateType &input);

 public:
  static StateType evalEncrypt(const LocalParamType &param);
  static StateType evalConstant(const LocalParamType &param);
  static StateType evalAdd(const StateType &lhs, const StateType &rhs);
  static StateType evalMul(const LocalParamType &resultParam,
                           const StateType &lhs, const StateType &rhs);
  static StateType evalRelinearize(const LocalParamType &inputParam,
                                   const StateType &input);
  static StateType evalModReduce(const LocalParamType &inputParam,
                                 const StateType &input);

  // logTotal: log(Ql / 2)
  // logBound: bound on ||m + t * e|| predicted by the model
  // logBudget: logTotal - logBound
  // as ||m + t * e|| < Ql / 2 for correct decryption
  static double toLogBound(const LocalParamType &param, const StateType &noise);
  static std::string toLogBoundString(const LocalParamType &param,
                                      const StateType &noise);
  static double toLogBudget(const LocalParamType &param,
                            const StateType &noise);
  static std::string toLogBudgetString(const LocalParamType &param,
                                       const StateType &noise);
  static double toLogTotal(const LocalParamType &param);
  static std::string toLogTotalString(const LocalParamType &param);
};

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYVARIANCECOEFFMODEL_H_
