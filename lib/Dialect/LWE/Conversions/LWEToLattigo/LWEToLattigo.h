#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOLATTIGO_LWETOLATTIGO_H_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOLATTIGO_LWETOLATTIGO_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir::heir::lwe {

#define GEN_PASS_DECL
#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h.inc"

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOLATTIGO_LWETOLATTIGO_H_
