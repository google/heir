#ifndef LIB_PARAMETERS_CKKS_PARAMS_H_
#define LIB_PARAMETERS_CKKS_PARAMS_H_

#include <cstdint>
#include <vector>

#include "lib/Parameters/RLWEParams.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

// Parameter for CKKS scheme at ModuleOp level
class SchemeParam : public RLWESchemeParam {
 public:
  SchemeParam(const RLWESchemeParam &rlweSchemeParam, int logDefaultScale)
      : RLWESchemeParam(rlweSchemeParam), logDefaultScale(logDefaultScale) {}

 private:
  // log of the default scale used to scale the message
  int logDefaultScale;

 public:
  int64_t getLogDefaultScale() const { return logDefaultScale; }
  void print(llvm::raw_ostream &os) const override;

  static SchemeParam getConcreteSchemeParam(std::vector<double> logqi,
                                            int logDefaultScale);
};

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_CKKS_PARAMS_H_
