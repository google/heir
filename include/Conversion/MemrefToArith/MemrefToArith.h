#ifndef HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_H_
#define HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "include/Conversion/MemrefToArith/MemrefToArith.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // HEIR_INCLUDE_CONVERSION_MEMREFTOARITH_MEMREFTOARITH_H_
