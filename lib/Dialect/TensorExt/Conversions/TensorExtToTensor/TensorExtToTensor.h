#ifndef LIB_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_
#define LIB_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::tensor_ext {

#define GEN_PASS_DECL
#include "lib/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"

}  // namespace mlir::heir::tensor_ext

#endif  // LIB_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_
