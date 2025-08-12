#ifndef LIB_ANALYSIS_NOISEANALYSIS_BFV_NOISECANEMBMODEL_H_
#define LIB_ANALYSIS_NOISEANALYSIS_BFV_NOISECANEMBMODEL_H_

#include <cassert>
#include <string>

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "lib/Parameters/BGV/Params.h"

namespace mlir {
namespace heir {
namespace bfv {

// canonical embedding noise model adapted
// from MMLGA22 https://eprint.iacr.org/2022/706
// from BMCM23 table 5 https://eprint.iacr.org/2023/600
// and from BGV/NoiseCanEmbModel
class NoiseCanEmbModel {
 public:
  // NoiseState stores the bound log2||e||^{can} for error e.
  using StateType = NoiseState;
  using SchemeParamType = bgv::SchemeParam;
  using LocalParamType = bgv::LocalParam;

 private:
  double getVarianceErr(const LocalParamType& param) const;
  double getVarianceKey(const LocalParamType& param) const;
  double getBScale(const LocalParamType& param) const;
  double getBKs(const LocalParamType& param) const;
  double getAssuranceFactor(const LocalParamType& param) const;
  double getPhi(const LocalParamType& param) const;
  double getRingExpansionFactor(const LocalParamType& param) const;

  StateType evalEncryptPk(const LocalParamType& param) const;
  StateType evalEncryptSk(const LocalParamType& param) const;
  StateType evalRelinearizeHYBRID(const LocalParamType& inputParam,
                                  const StateType& input) const;

 public:
  StateType evalEncrypt(const LocalParamType& param) const;
  StateType evalConstant(const LocalParamType& param) const;
  StateType evalAdd(const StateType& lhs, const StateType& rhs) const;
  StateType evalMul(const LocalParamType& resultParam, const StateType& lhs,
                    const StateType& rhs) const;
  StateType evalRelinearize(const LocalParamType& inputParam,
                            const StateType& input) const;
  StateType evalModReduce(const LocalParamType& inputParam,
                          const StateType& input) const;

  // logTotal: log(Ql / 2)
  // logBound: bound on ||m + t * e|| predicted by the model
  // logBudget: logTotal - logBound
  // as ||m + t * e|| < Ql / 2 for correct decryption
  double toLogBound(const LocalParamType& param, const StateType& noise) const;
  double toLogBudget(const LocalParamType& param, const StateType& noise) const;
  double toLogTotal(const LocalParamType& param) const;
};

}  // namespace bfv
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_NOISEANALYSIS_BFV_NOISECANEMBMODEL_H_
