#ifndef HEIR_LIB_DIALECT_SECRET_TRANSFORMS_CONSTANTPASSTHROUGHGENERIC_H_
#define HEIR_LIB_DIALECT_SECRET_TRANSFORMS_CONSTANTPASSTHROUGHGENERIC_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DECL_SECRETGENERICABSORBCONSTANTS
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

}  // namespace secret
}  // namespace heir
}  // namespace mlir

#endif  // HEIR_LIB_DIALECT_SECRET_TRANSFORMS_CONSTANTPASSTHROUGHGENERIC_H_
