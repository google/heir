#ifndef LIB_DIALECT_CKKS_SCIFRCKKS_CORNAMI_ANALYSIS_PASSES_H_
#define LIB_DIALECT_CKKS_SCIFRCKKS_CORNAMI_ANALYSIS_PASSES_H_

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/SCIFRCkks/Transforms/CKKSEstimator.h"
#include "lib/Dialect/SCIFRCkks/Transforms/OpenfheEstimator.h"

namespace mlir {
namespace cornami {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_CKKSESTIMATOR
#define GEN_PASS_DECL_OPENFHEESTIMATOR
#include "lib/Dialect/SCIFRCkks/Transforms/Passes.h.inc"

}  // namespace cornami
}  // namespace mlir

#endif  // LIB_DIALECT_CKKS_SCIFRCKKS_CORNAMI_ANALYSIS_PASSES_H_
