#ifndef LIB_PARAMETERS_BGV_PARAMS_H_
#define LIB_PARAMETERS_BGV_PARAMS_H_

#include <cstdint>
#include <vector>

#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Parameters/RLWEParams.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace bgv {

// Parameter for BGV scheme at ModuleOp level
class SchemeParam : public RLWESchemeParam {
 public:
  SchemeParam(const RLWESchemeParam& rlweSchemeParam, int64_t plaintextModulus)
      : RLWESchemeParam(rlweSchemeParam), plaintextModulus(plaintextModulus) {}

 private:
  // the plaintext modulus for BGV
  int64_t plaintextModulus;

 public:
  int64_t getPlaintextModulus() const { return plaintextModulus; }
  void print(llvm::raw_ostream& os) const override;

  static SchemeParam getConservativeSchemeParam(
      int level, int64_t plaintextModulus, int slotNumber, bool usePublicKey,
      bool encryptionTechniqueExtended);

  static SchemeParam getConcreteSchemeParam(std::vector<double> logqi,
                                            int64_t plaintextModulus,
                                            int slotNumber, bool usePublicKey,
                                            bool encryptionTechniqueExtended);

  static SchemeParam getSchemeParamFromAttr(SchemeParamAttr attr);
};

// Parameter for each BGV ciphertext SSA value.
class LocalParam : public RLWELocalParam {
 public:
  LocalParam(const SchemeParam* schemeParam, int currentLevel, int dimension)
      : RLWELocalParam(schemeParam, currentLevel, dimension) {}

 public:
  const SchemeParam* getSchemeParam() const {
    return static_cast<const SchemeParam*>(schemeParam);
  }
};

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_BGV_PARAMS_H_
