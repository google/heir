#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_CKKS_NOISEBYVARIANCECOEFFMODEL_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_CKKS_NOISEBYVARIANCECOEFFMODEL_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <string>

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Parameters/CKKS/Params.h"

namespace mlir {
namespace heir {
namespace ckks {

// coefficient embedding noise model using variance
class NoiseByVarianceCoeffModel {
 public:
  // NoiseState stores the variance var for the one coefficient of
  // the error 'e', assuming coefficients are IID.
  //
  // MP24/CCH+23 states that for two polynomial multipication, the variance of
  // one coefficient of the result can be approximated by ringDim * var_0 *
  // var_1, because the polynomial multipication is a convolution.
  using StateType = NoiseState;
  using SchemeParamType = ckks::SchemeParam;
  using LocalParamType = ckks::LocalParam;

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
                           const LocalParamType &lhsParam,
                           const LocalParamType &rhsParam, const StateType &lhs,
                           const StateType &rhs);
  static StateType evalRelinearize(const LocalParamType &inputParam,
                                   const StateType &input);

  // logTotal: log(Q / (t * 2))
  // logBound: bound on ||e|| predicted by the model
  // logBudget: logTotal - logBound
  // as ||e|| < Q / (t * 2) for correct decryption
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

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_CKKS_NOISEBYVARIANCECOEFFMODEL_H_
