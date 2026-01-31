#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Parameters/RLWEParams.h"

namespace mlir {
namespace heir {
namespace ckks {

SchemeParam getSchemeParamFromAttr(SchemeParamAttr attr) {
  auto logN = attr.getLogN();
  auto ringDim = pow(2, logN);
  auto Q = attr.getQ();
  auto P = attr.getP();
  auto logDefaultScale = attr.getLogDefaultScale();
  std::vector<int64_t> qiImpl;
  std::vector<int64_t> piImpl;
  std::vector<double> logqi;
  std::vector<double> logpi;
  for (auto qi : Q.asArrayRef()) {
    qiImpl.push_back(qi);
    logqi.push_back(log2(qi));
  }
  for (auto pi : P.asArrayRef()) {
    piImpl.push_back(pi);
    logpi.push_back(log2(pi));
  }
  auto level = logqi.size() - 1;
  auto dnum = ceil(static_cast<double>(qiImpl.size()) / piImpl.size());
  auto usePublicKey = attr.getEncryptionType() == CKKSEncryptionType::pk;
  auto encryptionTechniqueExtended =
      attr.getEncryptionTechnique() == CKKSEncryptionTechnique::extended;
  return SchemeParam(
      RLWESchemeParam(ringDim, level, logqi, qiImpl, dnum, logpi, piImpl,
                      usePublicKey, encryptionTechniqueExtended),
      logDefaultScale);
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
