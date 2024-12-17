#ifndef LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_H_
#define LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::ckks {

#define GEN_PASS_DECL
#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h.inc"

}  // namespace mlir::heir::ckks

#endif  // LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_H_
