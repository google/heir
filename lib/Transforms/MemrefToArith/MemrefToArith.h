#ifndef LIB_TRANSFORMS_MEMREFTOARITH_MEMREFTOARITH_H_
#define LIB_TRANSFORMS_MEMREFTOARITH_MEMREFTOARITH_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "lib/Transforms/MemrefToArith/MemrefToArith.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_MEMREFTOARITH_MEMREFTOARITH_H_
