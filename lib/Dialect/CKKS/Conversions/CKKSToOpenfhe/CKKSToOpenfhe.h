#ifndef LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOOPENFHE_CKKSTOOPENFHE_H_
#define LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOOPENFHE_CKKSTOOPENFHE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::ckks {

#define GEN_PASS_DECL
#include "lib/Dialect/CKKS/Conversions/CKKSToOpenfhe/CKKSToOpenfhe.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CKKS/Conversions/CKKSToOpenfhe/CKKSToOpenfhe.h.inc"

}  // namespace mlir::heir::ckks

#endif  // LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOOPENFHE_CKKSTOOPENFHE_H_
