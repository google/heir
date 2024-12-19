#ifndef LIB_DIALECT_ARITH_TRANSFORMS_QUARTERWIDEINT_H_
#define LIB_DIALECT_ARITH_TRANSFORMS_QUARTERWIDEINT_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace arith {

#define GEN_PASS_DECL_QUARTERWIDEINT
#include "lib/Dialect/Arith/Transforms/Passes.h.inc"

}  // namespace arith
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ARITH_TRANSFORMS_QUARTERWIDEINT_H_
