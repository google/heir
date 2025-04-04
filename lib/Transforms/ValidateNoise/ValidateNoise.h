#ifndef LIB_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_H_
#define LIB_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_H_

#include "mlir/include/mlir/Pass/Pass.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/ValidateNoise/ValidateNoise.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/ValidateNoise/ValidateNoise.h.inc"

constexpr StringRef kArgNoiseBoundAttrName = "noise.bound";

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_H_
