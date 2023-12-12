#ifndef INCLUDE_DIALECT_CGGI_TRANSFORMS_PASSES_H_
#define INCLUDE_DIALECT_CGGI_TRANSFORMS_PASSES_H_

#include "include/Dialect/CGGI/IR/CGGIDialect.h"
#include "include/Dialect/CGGI/Transforms/SetDefaultParameters.h"
#include "include/Dialect/CGGI/Transforms/StraightLineVectorizer.h"

namespace mlir {
namespace heir {
namespace cggi {

#define GEN_PASS_REGISTRATION
#include "include/Dialect/CGGI/Transforms/Passes.h.inc"

}  // namespace cggi
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_DIALECT_CGGI_TRANSFORMS_PASSES_H_
