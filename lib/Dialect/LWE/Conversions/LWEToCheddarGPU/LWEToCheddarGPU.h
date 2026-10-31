#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDARGPU_LWETOCHEDDARGPU_H_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDARGPU_LWETOCHEDDARGPU_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::lwe {

#define GEN_PASS_DECL
#include "lib/Dialect/LWE/Conversions/LWEToCheddarGPU/LWEToCheddarGPU.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/LWE/Conversions/LWEToCheddarGPU/LWEToCheddarGPU.h.inc"

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDARGPU_LWETOCHEDDARGPU_H_
