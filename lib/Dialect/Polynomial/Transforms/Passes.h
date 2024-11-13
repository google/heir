#ifndef LIB_DIALECT_POLYNOMAIL_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_POLYNOMAIL_TRANSFORMS_PASSES_H_

#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/Polynomial/Transforms/NTTRewrites.h"

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMAIL_TRANSFORMS_PASSES_H_
