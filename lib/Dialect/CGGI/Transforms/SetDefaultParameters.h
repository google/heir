#ifndef LIB_DIALECT_CGGI_TRANSFORMS_SETDEFAULTPARAMETERS_H_
#define LIB_DIALECT_CGGI_TRANSFORMS_SETDEFAULTPARAMETERS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_DECL_SETDEFAULTPARAMETERS
#include "lib/Dialect/CGGI/Transforms/Passes.h.inc"

}  // namespace cggi
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CGGI_TRANSFORMS_SETDEFAULTPARAMETERS_H_
