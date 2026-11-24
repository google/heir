#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOSTABLEHLO_LWETOSTABLEHLO_H_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOSTABLEHLO_LWETOSTABLEHLO_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep

namespace mlir::heir::lwe {

#define GEN_PASS_DECL
#include "lib/Dialect/LWE/Conversions/LWEToStableHLO/LWEToStableHLO.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/LWE/Conversions/LWEToStableHLO/LWEToStableHLO.h.inc"

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOSTABLEHLO_LWETOSTABLEHLO_H_
