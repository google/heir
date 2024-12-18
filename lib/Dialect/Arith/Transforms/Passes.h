#ifndef LIB_DIALECT_ARITH_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_ARITH_TRANSFORMS_PASSES_H_

#include "lib/Dialect/Arith/Transforms/QuarterWideInt.h"

namespace mlir {
namespace heir {
namespace arith {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Arith/Transforms/Passes.h.inc"

}  // namespace arith
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ARITH_TRANSFORMS_PASSES_H_
