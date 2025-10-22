#ifndef LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_ORIONTOCKKS_H_
#define LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_ORIONTOCKKS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::orion {

#define GEN_PASS_DECL
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h.inc"

}  // namespace mlir::heir::orion

#endif  // LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_ORIONTOCKKS_H_
