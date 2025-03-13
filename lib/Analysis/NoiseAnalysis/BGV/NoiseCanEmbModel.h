#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISECANEMBMODEL_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISECANEMBMODEL_H_

#include <cassert>
#include <string>

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Parameters/BGV/Params.h"

namespace mlir {
namespace heir {
namespace bgv {

// canonical embedding noise model from MMLGA22
// see https://eprint.iacr.org/2022/706
class NoiseCanEmbModel {
 public:
  // for MMLGA22, NoiseState stores the bound ||m + t * e||^{can} for error e.
  using StateType = NoiseState;
  using SchemeParamType = bgv::SchemeParam;
  using LocalParamType = bgv::LocalParam;

 private:
  static double getVarianceErr(const LocalParamType &param);
  static double getVarianceKey(const LocalParamType &param);
  static double getBScale(const LocalParamType &param);
  static double getBKs(const LocalParamType &param);
  static double getAssuranceFactor(const LocalParamType &param);
  static double getPhi(const LocalParamType &param);
  static double getRingExpansionFactor(const LocalParamType &param);

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

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISECANEMBMODEL_H_
