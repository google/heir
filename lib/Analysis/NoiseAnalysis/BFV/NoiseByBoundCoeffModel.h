#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_BFV_NOISEBYBOUNDCOEFFMODEL_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_BFV_NOISEBYBOUNDCOEFFMODEL_H_

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Parameters/BGV/Params.h"

namespace mlir {
namespace heir {
namespace bfv {

// Coefficient embedding noise model. Both average-case (from HPS19/KPZ21) and
// worst-case variants are supported.
class NoiseByBoundCoeffModel {
 public:
  NoiseByBoundCoeffModel(NoiseModelVariant variant) : variant(variant) {}
  ~NoiseByBoundCoeffModel() = default;

  // for KPZ21, NoiseState stores the bound log2||e|| for error e.
  using StateType = NoiseState;
  using SchemeParamType = bgv::SchemeParam;
  using LocalParamType = bgv::LocalParam;

 private:
  double getExpansionFactor(const LocalParamType &param) const;
  double getBoundErr(const LocalParamType &param) const;
  double getBoundKey(const LocalParamType &param) const;

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

 private:
  NoiseModelVariant variant;
};

}  // namespace bfv
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_BFV_NOISEBYBOUNDCOEFFMODEL_H_
