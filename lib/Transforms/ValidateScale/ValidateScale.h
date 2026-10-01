#ifndef LIB_TRANSFORMS_VALIDATESCALE_VALIDATESCALE_H_
#define LIB_TRANSFORMS_VALIDATESCALE_VALIDATESCALE_H_

#include "mlir/include/mlir/Pass/Pass.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/ValidateScale/ValidateScale.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/ValidateScale/ValidateScale.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_VALIDATESCALE_VALIDATESCALE_H_
