#ifndef LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_H_

#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_H_
