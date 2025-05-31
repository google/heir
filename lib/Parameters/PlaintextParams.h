#ifndef LIB_PARAMETERS_PLAINTEXTPARAMS_H_
#define LIB_PARAMETERS_PLAINTEXTPARAMS_H_

#include <cstdint>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

// Parameter for Plaintext backend ModuleOp level. This should only be used for
// the plaintext pipeline, which is for debugging programs.
// Scheme param and local param track the same thing: the default scale and
// incremental change in scale by applying a mod_reduce op.
//
// E.g., if the default log scale is 4, then each input plaintext is encoded
// with scale 16, and multiplying two plaintext will give a plaintext at scale
// 32, which is mod_reduced back to 16.
class PlaintextSchemeParam {
 public:
  PlaintextSchemeParam(int defaultLogScale)
      : defaultLogScale(defaultLogScale) {}
  PlaintextSchemeParam(const PlaintextSchemeParam *other, int level,
                       int dimension)
      : defaultLogScale(other->defaultLogScale) {}

 private:
  // log of the default scale used to scale the message
  int64_t defaultLogScale;

 public:
  int64_t getDefaultLogScale() const { return defaultLogScale; }
  void print(llvm::raw_ostream &os) const;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_PLAINTEXTPARAMS_H_
