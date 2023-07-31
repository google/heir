#ifndef INCLUDE_DIALECT_SECRET_TRANSFORMS_PASSES_H_
#define INCLUDE_DIALECT_SECRET_TRANSFORMS_PASSES_H_

#include "include/Dialect/Secret/IR/SecretDialect.h"
#include "include/Dialect/Secret/Transforms/ForgetSecrets.h"

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_REGISTRATION
#include "include/Dialect/Secret/Transforms/Passes.h.inc"

}  // namespace secret
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_DIALECT_SECRET_TRANSFORMS_PASSES_H_
