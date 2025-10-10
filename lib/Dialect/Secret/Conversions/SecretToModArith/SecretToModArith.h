#ifndef LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOMODARITH_SECRETTOMODARITH_H_
#define LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOMODARITH_SECRETTOMODARITH_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::mod_arith {

#define GEN_PASS_DECL
#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h.inc"

}  // namespace mlir::heir::mod_arith

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOMODARITH_SECRETTOMODARITH_H_
