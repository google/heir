#ifndef LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_ORIONTOCKKS_H_
#define LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_ORIONTOCKKS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
// IWYU pragma: end_keep

namespace mlir::heir::orion {

#define GEN_PASS_DECL
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h.inc"

}  // namespace mlir::heir::orion

#endif  // LIB_DIALECT_ORION_CONVERSIONS_ORIONTOCKKS_ORIONTOCKKS_H_
