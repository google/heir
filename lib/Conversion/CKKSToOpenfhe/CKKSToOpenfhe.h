#ifndef LIB_CONVERSION_CKKSTOOPENFHE_CKKSTOOPENFHE_H_
#define LIB_CONVERSION_CKKSTOOPENFHE_CKKSTOOPENFHE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::ckks {

#define GEN_PASS_DECL
#include "lib/Conversion/CKKSToOpenfhe/CKKSToOpenfhe.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/CKKSToOpenfhe/CKKSToOpenfhe.h.inc"

}  // namespace mlir::heir::ckks

#endif  // LIB_CONVERSION_CKKSTOOPENFHE_CKKSTOOPENFHE_H_
