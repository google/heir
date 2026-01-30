#ifndef LIB_DIALECT_CGGI_CORNAMI_ANALYSIS_PASSES_H_
#define LIB_DIALECT_CGGI_CORNAMI_ANALYSIS_PASSES_H_

#include "lib/Dialect/SCIFRBool/Transforms/CGGIEstimator.h"

namespace mlir {
namespace cornami {
namespace scifrbool {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_CGGIESTIMATOR
#include "lib/Dialect/SCIFRBool/Transforms/Passes.h.inc"

}  // namespace scifrbool
}  // namespace cornami
}  // namespace mlir

#endif  // LIB_DIALECT_CGGI_CORNAMI_ANALYSIS_PASSES_H_
