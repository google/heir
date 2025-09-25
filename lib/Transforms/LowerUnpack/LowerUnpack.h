#ifndef LIB_TRANSFORMS_LOWERUNPACK_LOWERUNPACK_H_
#define LIB_TRANSFORMS_LOWERUNPACK_LOWERUNPACK_H_

#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"           // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/LowerUnpack/LowerUnpack.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/LowerUnpack/LowerUnpack.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LOWERUNPACK_LOWERUNPACK_H_
