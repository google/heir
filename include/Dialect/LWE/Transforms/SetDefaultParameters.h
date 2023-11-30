#ifndef INCLUDE_DIALECT_LWE_TRANSFORMS_SETDEFAULTPARAMETERS_H_
#define INCLUDE_DIALECT_LWE_TRANSFORMS_SETDEFAULTPARAMETERS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DECL_SETDEFAULTPARAMETERS
#include "include/Dialect/LWE/Transforms/Passes.h.inc"

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_DIALECT_LWE_TRANSFORMS_SETDEFAULTPARAMETERS_H_
