#include "lib/Parameters/BGV/Params.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Parameters/RLWEParams.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace bgv {

SchemeParam SchemeParam::getConservativeSchemeParam(
    int level, int64_t plaintextModulus, int slotNumber, bool usePublicKey,
    bool encryptionTechniqueExtended) {
  // Use only half of the BGV slot number to make 1-dim vector.
  return SchemeParam(
      RLWESchemeParam::getConservativeRLWESchemeParam(
          level, 2 * slotNumber, usePublicKey, encryptionTechniqueExtended),
      plaintextModulus);
}

SchemeParam SchemeParam::getConcreteSchemeParam(
    std::vector<double> logqi, int64_t plaintextModulus, int slotNumber,
    bool usePublicKey, bool encryptionTechniqueExtended) {
  // Use only half of the BGV slot number to make 1-dim vector.
  return SchemeParam(RLWESchemeParam::getConcreteRLWESchemeParam(
                         std::move(logqi), 2 * slotNumber, usePublicKey,
                         encryptionTechniqueExtended, plaintextModulus),
                     plaintextModulus);
}

SchemeParam SchemeParam::getSchemeParamFromAttr(SchemeParamAttr attr) {
  auto logN = attr.getLogN();
  auto ringDim = pow(2, logN);
  auto plaintextModulus = attr.getPlaintextModulus();
  auto Q = attr.getQ();
  auto P = attr.getP();
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
  auto usePublicKey = attr.getEncryptionType() == BGVEncryptionType::pk;
  auto encryptionTechniqueExtended =
      attr.getEncryptionTechnique() == BGVEncryptionTechnique::extended;
  return SchemeParam(
      RLWESchemeParam(ringDim, level, logqi, qiImpl, dnum, logpi, piImpl,
                      usePublicKey, encryptionTechniqueExtended),
      plaintextModulus);
}

void SchemeParam::print(llvm::raw_ostream& os) const {
  os << "plaintextModulus: " << plaintextModulus << "\n";
  RLWESchemeParam::print(os);
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
