#ifndef LIB_DIALECT_CKKS_SCIFRCKKS_CORNAMI_ANALYSIS_PASSES_H_
#define LIB_DIALECT_CKKS_SCIFRCKKS_CORNAMI_ANALYSIS_PASSES_H_

#include "lib/Backend/cornami/Dialect/SCIFRCkks/Analysis/CKKSEstimator.h"
#include "lib/Backend/cornami/Dialect/SCIFRCkks/Analysis/OpenfheEstimator.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"

namespace mlir {
namespace cornami {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_CKKSESTIMATOR
#define GEN_PASS_DECL_OPENFHEESTIMATOR
#include "lib/Backend/cornami/Dialect/SCIFRCkks/Analysis/Passes.h.inc"

}  // namespace cornami
}  // namespace mlir

#endif  // LIB_DIALECT_CKKS_SCIFRCKKS_CORNAMI_ANALYSIS_PASSES_H_
