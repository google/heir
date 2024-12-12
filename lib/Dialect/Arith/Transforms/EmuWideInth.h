#ifndef LIB_DIALECT_ARITH_TRANSFORMS_EMUWIDEINTH_H_
#define LIB_DIALECT_ARITH_TRANSFORMS_EMUWIDEINTH_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace arith {

#define GEN_PASS_DECL_EMUWIDEINTH
#include "lib/Dialect/Arith/Transforms/Passes.h.inc"

}  // namespace arith
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ARITH_TRANSFORMS_EMUWIDEINTH_H_
