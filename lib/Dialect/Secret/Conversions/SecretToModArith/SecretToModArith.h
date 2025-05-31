#ifndef LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOModArith_SECRETTOModArith_H_
#define LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOModArith_SECRETTOModArith_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DECL
#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h.inc"

}  // namespace mlir::heir

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOModArith_SECRETTOModArith_H_
