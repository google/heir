#ifndef LIB_DIALECT_CGGI_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_CGGI_TRANSFORMS_PASSES_H_

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/Transforms/BooleanLineVectorizer.h"
#include "lib/Dialect/CGGI/Transforms/SetDefaultParameters.h"

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CGGI/Transforms/Passes.h.inc"

}  // namespace cggi
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CGGI_TRANSFORMS_PASSES_H_
