#include "lib/Parameters/CKKS/Params.h"

#include <cassert>
#include <utility>
#include <vector>

#include "lib/Parameters/RLWEParams.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

SchemeParam SchemeParam::getConcreteSchemeParam(
    std::vector<double> logqi, int logDefaultScale, int slotNumber,
    bool usePublicKey, bool encryptionTechniqueExtended) {
  // CKKS slot number = ringDim / 2
  return SchemeParam(RLWESchemeParam::getConcreteRLWESchemeParam(
                         std::move(logqi), 2 * slotNumber, usePublicKey,
                         encryptionTechniqueExtended),
                     logDefaultScale);
}

SchemeParam SchemeParam::getSchemeParamFromAttr(SchemeParamAttr attr) {
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

void SchemeParam::print(llvm::raw_ostream& os) const {
  os << "logDefaultScale: " << logDefaultScale << "\n";
  RLWESchemeParam::print(os);
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
