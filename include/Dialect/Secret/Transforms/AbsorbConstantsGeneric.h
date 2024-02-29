#ifndef HEIR_INCLUDE_DIALECT_SECRET_TRANSFORMS_ABSORBCONSTANTSGENERIC_H_
#define HEIR_INCLUDE_DIALECT_SECRET_TRANSFORMS_ABSORBCONSTANTSGENERIC_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DECL_SECRETGENERICABSORBCONSTANTS
#include "include/Dialect/Secret/Transforms/Passes.h.inc"

}  // namespace secret
}  // namespace heir
}  // namespace mlir

#endif  // HEIR_INCLUDE_DIALECT_SECRET_TRANSFORMS_ABSORBCONSTANTSGENERIC_H_
