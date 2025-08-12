#ifndef LIB_PARAMETERS_CKKS_PARAMS_H_
#define LIB_PARAMETERS_CKKS_PARAMS_H_

#include <cstdint>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Parameters/RLWEParams.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

// Parameter for CKKS scheme at ModuleOp level
class SchemeParam : public RLWESchemeParam {
 public:
  SchemeParam(const RLWESchemeParam& rlweSchemeParam, int logDefaultScale)
      : RLWESchemeParam(rlweSchemeParam), logDefaultScale(logDefaultScale) {}

 private:
  // log of the default scale used to scale the message
  int logDefaultScale;

 public:
  int64_t getLogDefaultScale() const { return logDefaultScale; }
  void print(llvm::raw_ostream& os) const override;

  static SchemeParam getConcreteSchemeParam(std::vector<double> logqi,
                                            int logDefaultScale, int slotNumber,
                                            bool usePublicKey,
                                            bool encryptionTechniqueExtended);

  static SchemeParam getSchemeParamFromAttr(SchemeParamAttr attr);
};

// Parameter for each SSA ciphertext SSA value.
class LocalParam : public RLWELocalParam {
 public:
  LocalParam(const SchemeParam* schemeParam, int currentLevel, int dimension)
      : RLWELocalParam(schemeParam, currentLevel, dimension) {}

 public:
  const SchemeParam* getSchemeParam() const {
    return static_cast<const SchemeParam*>(schemeParam);
  }
};

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_CKKS_PARAMS_H_
