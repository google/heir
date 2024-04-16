#ifndef INCLUDE_DIALECT_BGV_TRANSFORMS_PASSES_H_
#define INCLUDE_DIALECT_BGV_TRANSFORMS_PASSES_H_

#include "include/Dialect/BGV/IR/BGVDialect.h"
#include "include/Dialect/BGV/Transforms/AddClientInterface.h"

namespace mlir {
namespace heir {
namespace bgv {

#define GEN_PASS_REGISTRATION
#include "include/Dialect/BGV/Transforms/Passes.h.inc"

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_DIALECT_BGV_TRANSFORMS_PASSES_H_
