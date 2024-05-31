#ifndef LIB_DIALECT_ARITHEXT_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_ARITHEXT_TRANSFORMS_PASSES_H_

#include "lib/Dialect/ArithExt/IR/ArithExtDialect.h"
#include "lib/Dialect/ArithExt/Transforms/RemToArithExt.h"

namespace mlir {
namespace heir {
namespace arith_ext {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/ArithExt/Transforms/Passes.h.inc"

}  // namespace arith_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ARITHEXT_TRANSFORMS_PASSES_H_
