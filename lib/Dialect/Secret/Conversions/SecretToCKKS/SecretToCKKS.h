#ifndef LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCKKS_SECRETTOCKKS_H_
#define LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCKKS_SECRETTOCKKS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DECL
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h.inc"

}  // namespace mlir::heir

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCKKS_SECRETTOCKKS_H_
