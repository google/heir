#ifndef LIB_DIALECT_MODARITH_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_MODARITH_TRANSFORMS_PASSES_H_

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/Transforms/ConvertToMac.h"

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/ModArith/Transforms/Passes.h.inc"

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MODARITH_TRANSFORMS_PASSES_H_
