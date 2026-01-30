#ifndef LIB_DIALECT_CGGI_CORNAMI_ANALYSIS_PASSES_H_
#define LIB_DIALECT_CGGI_CORNAMI_ANALYSIS_PASSES_H_

#include "lib/Backend/cornami/Dialect/SCIFRBool/Analysis/CGGIEstimator.h"

namespace mlir {
namespace cornami {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_CGGIESTIMATOR
#include "lib/Backend/cornami/Dialect/SCIFRBool/Analysis/Passes.h.inc"

}  // namespace cornami
}  // namespace mlir

#endif  // LIB_DIALECT_CGGI_CORNAMI_ANALYSIS_PASSES_H_
