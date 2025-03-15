#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYBOUNDCOEFFMODEL_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYBOUNDCOEFFMODEL_H_

#include <cassert>
#include <string>

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Parameters/BGV/Params.h"

namespace mlir {
namespace heir {
namespace bgv {

// coefficient embedding noise model
// both average-case (from HPS19/KPZ21) and worst-case
// use template here just for the sake of code reuse
// W for worst-case
template <bool W>
class NoiseByBoundCoeffModel {
 public:
  // for KPZ21, NoiseState stores the bound ||e|| for error e
  // instead of t * ||e||.
  using StateType = NoiseState;
  using SchemeParamType = SchemeParam;
  using LocalParamType = LocalParam;

 private:
  static double getExpansionFactor(const LocalParamType &param);
  static double getBoundErr(const LocalParamType &param);
  static double getBoundKey(const LocalParamType &param);

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

// user-facing typedefs
using NoiseByBoundCoeffAverageCaseModel = NoiseByBoundCoeffModel<false>;
using NoiseByBoundCoeffWorstCaseModel = NoiseByBoundCoeffModel<true>;

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISEBYBOUNDCOEFFMODEL_H_
