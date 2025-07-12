#ifndef LIB_ANALYSIS_NOISEANALYSIS_BFV_NOISEBYVARIANCECOEFFMODEL_H_
#define LIB_ANALYSIS_NOISEANALYSIS_BFV_NOISEBYVARIANCECOEFFMODEL_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <string>

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Parameters/BGV/Params.h"

namespace mlir {
namespace heir {
namespace bfv {

// coefficient embedding noise model using variance
class NoiseByVarianceCoeffModel {
 public:
  // NoiseState stores the variance log2(var) for the one coefficient of
  // the error 'e', assuming coefficients are IID.
  // Note that BMCM23 tracks the invariant noise t/q(c0 + c1 * s).
  // We adapt it to track the 'e' in [(q/t)m + e]_q for convenience
  // and consistency with the other models.
  //
  // The main contribution of BMCM23 is that it considers the dependence between
  // random variables caused by 's', so the max degree T in 's^T' of the
  // invariant noise needs to be tracked.
  //
  // MP24/CCH+23 states that for two polynomial multiplication, the variance of
  // one coefficient of the result can be approximated by ringDim * var_0 *
  // var_1, because the polynomial multiplication is a convolution.
  using StateType = NoiseState;
  using SchemeParamType = bgv::SchemeParam;
  using LocalParamType = bgv::LocalParam;

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

  // logTotal: log(Q / (t * 2))
  // logBound: bound on ||e|| predicted by the model
  // logBudget: logTotal - logBound
  // as ||e|| < Q / (t * 2) for correct decryption
  double toLogBound(const LocalParamType &param, const StateType &noise) const;
  double toLogBudget(const LocalParamType &param, const StateType &noise) const;
  double toLogTotal(const LocalParamType &param) const;
};

}  // namespace bfv
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_NOISEANALYSIS_BFV_NOISEBYVARIANCECOEFFMODEL_H_
