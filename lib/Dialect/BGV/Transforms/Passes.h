#ifndef LIB_DIALECT_BGV_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_BGV_TRANSFORMS_PASSES_H_

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/Transforms/AddClientInterface.h"

namespace mlir {
namespace heir {
namespace bgv {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/BGV/Transforms/Passes.h.inc"

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_BGV_TRANSFORMS_PASSES_H_
