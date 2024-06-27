#ifndef LIB_DIALECT_SECRET_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_SECRET_TRANSFORMS_PASSES_H_

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/Transforms/CaptureGenericAmbientScope.h"
#include "lib/Dialect/Secret/Transforms/DistributeGeneric.h"
#include "lib/Dialect/Secret/Transforms/ExtractGenericBody.h"
#include "lib/Dialect/Secret/Transforms/ForgetSecrets.h"
#include "lib/Dialect/Secret/Transforms/GenericAbsorbConstants.h"
#include "lib/Dialect/Secret/Transforms/GenericAbsorbDealloc.h"
#include "lib/Dialect/Secret/Transforms/MergeAdjacentGenerics.h"

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

}  // namespace secret
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_SECRET_TRANSFORMS_PASSES_H_
